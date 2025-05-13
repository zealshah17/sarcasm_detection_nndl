import os
import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from models.sarcasm_detector import SarcasmDetector, SentenceEncoderSarcasmDetector
from utils.data_loader import load_irony_hq_dataset


def evaluate(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    test_dataloader = load_irony_hq_dataset(args.data_dir, batch_size=args.batch_size)
    
    # Create model
    if args.model_type == 'graph':
        print("Using GraphTransformer model")
        model = SarcasmDetector(
            sentence_encoder=args.sentence_encoder,
            hidden_dim=args.hidden_dim,
            graph_depth=args.graph_depth,
            edge_dim=args.edge_dim,
            max_seq_len=args.max_seq_len,
            device=device
        )
    else:
        print("Using simple SentenceEncoder model")
        model = SentenceEncoderSarcasmDetector(
            sentence_encoder=args.sentence_encoder,
            device=device
        )
    
    # Move model to device
    model = model.to(device)
    
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Evaluation mode
    model.eval()
    
    # Evaluate
    all_logits = []
    all_preds = []
    all_labels = []
    all_texts = []
    
    with torch.no_grad():
        test_bar = tqdm(test_dataloader, desc="Evaluating")
        for texts, labels in test_bar:
            # Move labels to device
            labels = labels.to(device)
            
            # Forward pass
            logits = model(texts)
            
            # Get predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Save results
            all_logits.extend(logits.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_texts.extend(texts)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_logits)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_logits)
    pr_auc = average_precision_score(all_labels, all_logits)
    
    # Get confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist()
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    roc_plot_path = os.path.join(args.output_dir, 'roc_curve.png')
    plt.savefig(roc_plot_path)
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    pr_plot_path = os.path.join(args.output_dir, 'pr_curve.png')
    plt.savefig(pr_plot_path)
    
    # Save incorrect predictions for analysis
    incorrect_indices = [i for i, (label, pred) in enumerate(zip(all_labels, all_preds)) if label != pred]
    incorrect_examples = []
    
    for idx in incorrect_indices[:50]:  # Limit to 50 examples
        incorrect_examples.append({
            'text': all_texts[idx],
            'true_label': int(all_labels[idx]),
            'predicted_label': int(all_preds[idx]),
            'confidence': float(np.abs(all_logits[idx]))
        })
    
    incorrect_path = os.path.join(args.output_dir, 'incorrect_predictions.json')
    with open(incorrect_path, 'w') as f:
        json.dump(incorrect_examples, f, indent=4)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sarcasm detection model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="graph", choices=["graph", "simple"], 
                        help="Model type: graph (GraphTransformer) or simple (SentenceEncoder)")
    parser.add_argument("--sentence_encoder", type=str, default="paraphrase-miniLM-l3-v2", 
                        help="Pre-trained sentence encoder model")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--graph_depth", type=int, default=6, help="Graph Transformer depth")
    parser.add_argument("--edge_dim", type=int, default=256, help="Edge feature dimension")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length")
    
    # Checkpoint argument
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./evaluation", help="Output directory")
    
    args = parser.parse_args()
    
    evaluate(args)