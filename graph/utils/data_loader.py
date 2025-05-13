import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split


class SarcasmDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.float)


def load_sarc_dataset(data_dir, max_samples=2000, test_size=0.2, random_state=42):
    """
    Load the SARC dataset and split into train and validation sets.
    
    Args:
        data_dir: Directory containing the dataset files
        max_samples: Maximum number of samples to use (default: 5000)
        test_size: Proportion of data to use for validation (default: 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        train_dataloader, val_dataloader: DataLoaders for training and validation
    """
    try:
        # Try loading the dataset file
        df = pd.read_csv(os.path.join(data_dir, 'headlines.csv'))
        
        # Check if we have the required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            print("Required columns 'text' and 'label' not found. Attempting to adapt.")
            
            # Try to adapt to different column names
            # This is a simple adaptation, you might need to modify based on actual dataset structure
            if 'comment' in df.columns:
                df = df.rename(columns={'comment': 'text'})
            
            if 'sarcastic' in df.columns:
                df = df.rename(columns={'sarcastic': 'label'})
            
            # If still missing columns, raise error
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("Required columns 'text' and 'label' not found and couldn't be adapted.")
    
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Error loading SARC dataset: {e}")
        print("Creating a synthetic dataset for development purposes.")
        
        # Create a synthetic dataset for development
        np.random.seed(random_state)
        
        # Generate synthetic data
        sarcastic_texts = [
            "Oh, I'm so happy to be stuck in traffic for two hours.",
            "Yeah, because everyone loves Mondays.",
            "What a GREAT idea, let's make things even more complicated.",
            "Sure, because that worked so well last time.",
            "Wow, you're really a genius for figuring that out."
        ] * 500
        
        non_sarcastic_texts = [
            "I enjoyed the movie we saw yesterday.",
            "The weather is beautiful today.",
            "This restaurant has great food.",
            "I'm looking forward to the weekend.",
            "The meeting went better than expected."
        ] * 500
        
        texts = sarcastic_texts + non_sarcastic_texts
        labels = [1] * len(sarcastic_texts) + [0] * len(non_sarcastic_texts)
        
        # Create dataframe
        df = pd.DataFrame({'text': texts, 'label': labels})
    
    # Limit to max_samples
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=random_state)
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])
    
    # Create datasets
    train_dataset = SarcasmDataset(train_df['text'].tolist(), train_df['label'].tolist())
    val_dataset = SarcasmDataset(val_df['text'].tolist(), val_df['label'].tolist())
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    
    return train_dataloader, val_dataloader


def load_irony_hq_dataset(data_dir, batch_size=32):
    """
    Load the IronyHQ dataset for evaluation.
    
    Args:
        data_dir: Directory containing the dataset files
        batch_size: Batch size for the DataLoader
        
    Returns:
        test_dataloader: DataLoader for testing
    """
    try:
        # Try loading the dataset file
        df = pd.read_csv(os.path.join(data_dir, 'irony_hq.csv'))
        
        # Check if we have the required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            print("Required columns 'text' and 'label' not found. Attempting to adapt.")
            
            # Try to adapt to different column names
            # This is a simple adaptation, you might need to modify based on actual dataset structure
            if 'tweet' in df.columns:
                df = df.rename(columns={'tweet': 'text'})
            
            if 'irony' in df.columns:
                df = df.rename(columns={'irony': 'label'})
            
            # If still missing columns, raise error
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("Required columns 'text' and 'label' not found and couldn't be adapted.")
    
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Error loading IronyHQ dataset: {e}")
        print("Creating a synthetic dataset for development purposes.")
        
        # Create a synthetic dataset for development
        
        # Generate synthetic data
        sarcastic_texts = [
            "Just what I needed today, a computer crash right before deadline.",
            "I love when people drive 20 mph under the speed limit.",
            "Nothing says 'productive day' like 5 hours of meetings.",
            "Wow, thanks for explaining that really obvious concept to me.",
            "I'm thrilled to be working on the weekend again."
        ] * 20
        
        non_sarcastic_texts = [
            "The new update has improved system performance.",
            "The team worked hard to meet the deadline.",
            "The study showed significant results.",
            "I appreciate your help with this project.",
            "The conference was very informative."
        ] * 20
        
        texts = sarcastic_texts + non_sarcastic_texts
        labels = [1] * len(sarcastic_texts) + [0] * len(non_sarcastic_texts)
        
        # Create dataframe
        df = pd.DataFrame({'text': texts, 'label': labels})
    
    # Create dataset
    test_dataset = SarcasmDataset(df['text'].tolist(), df['label'].tolist())
    
    # Create data loader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Loaded {len(test_dataset)} test samples from IronyHQ.")
    
    return test_dataloader