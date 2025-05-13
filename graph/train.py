import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm
import time
import json

from models.sarcasm_detector import SarcasmDetector, SentenceEncoderSarcasmDetector
from utils.data_loader import load_sarc_dataset


def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    train_dataloader, val_dataloader = load_sarc_dataset(
        args.data_dir, 
        max_samples=args.max_samples, 
        test_size=args.val_split
    )
    
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
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    training_stats = []
    early_stopping_patience = 2
    no_improvement_epochs = 0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        train_bar = tqdm(train_dataloader, desc="Training")
        for texts, labels in train_bar:
            # Move labels to device
            labels = labels.to(device)
            
            # Forward pass
            logits = model(texts)
            loss = criterion(logits, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            train_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_dataloader)
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='binary')
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, desc="Validation")
            for texts, labels in val_bar:
                # Move labels to device
                labels = labels.to(device)
                
                # Forward pass
                logits = model(texts)
                loss = criterion(logits, labels)
                
                # Update statistics
                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
                
                val_bar.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='binary')
        val_precision = precision_score(val_labels, val_preds, average='binary')
        val_recall = recall_score(val_labels, val_preds, average='binary')
        
        # Update learning rate scheduler
        scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            
            # Save model
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
            }, model_path)
            
            print(f"Best model saved with F1: {val_f1:.4f}")
        else:
            no_improvement_epochs += 1
            print(f"No improvement in F1 for {no_improvement_epochs} epoch(s)")

        # Early stopping check
        if no_improvement_epochs >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
        # Print statistics
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")
        print(f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}")
        
        # Save statistics
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        training_stats.append(epoch_stats)
    
    # Save training statistics
    stats_path = os.path.join(args.output_dir, 'training_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=4)
    
    print(f"\nTraining completed! Best model at epoch {best_epoch} with F1: {best_val_f1:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)
    
    print(f"Final model saved to {final_model_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sarcasm detection model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the dataset")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum number of samples to use")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="graph", choices=["graph", "simple"], 
                        help="Model type: graph (GraphTransformer) or simple (SentenceEncoder)")
    parser.add_argument("--sentence_encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Pre-trained sentence encoder model")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--graph_depth", type=int, default=6, help="Graph Transformer depth")
    parser.add_argument("--edge_dim", type=int, default=256, help="Edge feature dimension")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    
    args = parser.parse_args()
    
    train(args)