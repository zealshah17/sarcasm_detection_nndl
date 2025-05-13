import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from graph_transformer_pytorch import GraphTransformer


class SarcasmDetector(nn.Module):
    def __init__(
        self, 
        sentence_encoder='sentence-transformers/all-MiniLM-L6-v2',
        hidden_dim=128, 
        graph_depth=6,
        edge_dim=128,
        max_seq_len=128,
        device='cpu'
    ):
        super().__init__()
        
        self.device = device
        self.max_seq_len = max_seq_len
        
        # Load pre-trained tokenizer and model for sentence embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(sentence_encoder)
        self.sentence_encoder = AutoModel.from_pretrained(sentence_encoder, add_pooling_layer=False)
        
        # Embedding dimension from the pre-trained model
        self.emb_dim = self.sentence_encoder.config.hidden_size
        
        # Projection from sentence embeddings to graph node features
        self.node_projection = nn.Linear(self.emb_dim, hidden_dim)

        self.edge_dim = edge_dim
        
        # Edge feature generator
        self.edge_projection = nn.Sequential(
            nn.Linear(self.emb_dim * 2, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)
        )
        
        # Graph Transformer for learning graph representations
        self.graph_transformer = GraphTransformer(
            dim=hidden_dim,
            depth=graph_depth,
            edge_dim=edge_dim,
            with_feedforwards=True,
            gated_residual=True,
            rel_pos_emb=True
        )
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def encode_text(self, texts):
        """Encode a list of texts into embeddings using the sentence encoder"""
        # Tokenize texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                              max_length=self.max_seq_len, return_tensors="pt").to(self.device)
        
        # Get embeddings from sentence encoder
        with torch.no_grad():
            outputs = self.sentence_encoder(**inputs)
        
        # Use CLS token embedding as sentence representation
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        masked_output = outputs.last_hidden_state * attention_mask
        sum_embeddings = masked_output.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
        embeddings = sum_embeddings / sum_mask

        return embeddings
    
    def build_graph(self, texts):
        """Build graph nodes and edges from texts"""
        # Get sentence embeddings
        sentence_embeddings = self.encode_text(texts)
        batch_size = len(texts)
        
        # Create node features
        nodes = self.node_projection(sentence_embeddings)
        nodes = nodes.unsqueeze(1)  # Shape: [batch_size, 1, hidden_dim]
        
        # For single sentence classification, we use a simple approach:
        # 1. Split the sentence into tokens
        # 2. Create additional nodes from token embeddings
        tokens_batch = []
        for text in texts:
            tokens = text.split()[:self.max_seq_len-1]  # Limit tokens to max_seq_len-1
            tokens_batch.append(tokens)
        
        # Get token embeddings
        token_embeddings_list = []
        masks = []
        
        for i, tokens in enumerate(tokens_batch):
            if not tokens:  # Skip empty token lists
                token_embeddings_list.append(torch.zeros(1, self.emb_dim, device=self.device))
                masks.append(torch.tensor([True], device=self.device))
                continue
                
            # Tokenize tokens
            token_inputs = self.tokenizer(tokens, padding=True, truncation=True, return_tensors="pt").to(self.device)

            
            # Get token embeddings
            with torch.no_grad():
                token_outputs = self.sentence_encoder(**token_inputs)
            
            # Use CLS token embeddings as token representations
            attention_mask = token_inputs['attention_mask'].unsqueeze(-1)
            masked_output = token_outputs.last_hidden_state * attention_mask
            sum_embeddings = masked_output.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
            token_embeddings = sum_embeddings / sum_mask
        
            token_embeddings_list.append(token_embeddings)
            
            # Create mask
            mask = torch.ones(len(tokens), dtype=torch.bool, device=self.device)
            masks.append(mask)
        
        # Pad token embeddings and masks to the same length
        # max_tokens = max(len(emb) for emb in token_embeddings_list)
        max_tokens = max(mask.sum().item() for mask in masks)
        
        # Ensure we don't exceed max_seq_len
        max_tokens = min(max_tokens, self.max_seq_len - 1)
        
        padded_token_embeddings = []
        padded_masks = []
        
        for token_embeddings, mask in zip(token_embeddings_list, masks):
            # Pad token embeddings
            if len(token_embeddings) < max_tokens:
                padding = torch.zeros(max_tokens - len(token_embeddings), self.emb_dim, device=self.device)
                token_embeddings = torch.cat([token_embeddings, padding], dim=0)
            else:
                token_embeddings = token_embeddings[:max_tokens]
            
            padded_token_embeddings.append(token_embeddings)
            
            # Pad masks
            if len(mask) < max_tokens:
                padding = torch.zeros(max_tokens - len(mask), dtype=torch.bool, device=self.device)
                mask = torch.cat([mask, padding], dim=0)
            else:
                mask = mask[:max_tokens]
            
            padded_masks.append(mask)
        
        # Stack padded token embeddings and masks
        token_embeddings_tensor = torch.stack(padded_token_embeddings, dim=0)
        masks_tensor = torch.stack(padded_masks, dim=0)
        
        # Project token embeddings to hidden dimension
        token_nodes = self.node_projection(token_embeddings_tensor)
        
        # Combine sentence nodes and token nodes
        nodes = torch.cat([nodes, token_nodes], dim=1)  # Shape: [batch_size, 1+max_tokens, hidden_dim]
        
        # Create full mask including the sentence node
        sentence_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=self.device)
        full_mask = torch.cat([sentence_mask, masks_tensor], dim=1)
        
        # Create edges between nodes
        # edges = torch.zeros(batch_size, nodes.size(1), nodes.size(1), edge_dim, device=self.device)
        edges = torch.zeros(batch_size, nodes.size(1), nodes.size(1), self.edge_dim, device=self.device)
        
        # Connect sentence node with token nodes
        for b in range(batch_size):
            # Sentence embeddings repeated for each token
            sent_emb_repeated = sentence_embeddings[b].repeat(max_tokens, 1)
            
            # Token embeddings
            token_embs = token_embeddings_tensor[b]
            
            # Concatenate embeddings for edge features
            edge_feats_st = torch.cat([sent_emb_repeated, token_embs], dim=1)
            edge_feats_ts = torch.cat([token_embs, sent_emb_repeated], dim=1)
            
            # Project to edge features
            edge_st = self.edge_projection(edge_feats_st)
            edge_ts = self.edge_projection(edge_feats_ts)
            
            # Set edge features
            edges[b, 0, 1:] = edge_st
            edges[b, 1:, 0] = edge_ts
            
            # Create edges between token nodes (fully connected)
            for i in range(max_tokens):
                for j in range(max_tokens):
                    if i != j:
                        token_i = token_embeddings_tensor[b, i]
                        token_j = token_embeddings_tensor[b, j]
                        edge_feat = torch.cat([token_i, token_j], dim=0)
                        edges[b, i+1, j+1] = self.edge_projection(edge_feat)
        
        return nodes, edges, full_mask
    
    def forward(self, texts):
        # Build graph from texts
        nodes, edges, mask = self.build_graph(texts)
        
        # Apply Graph Transformer
        nodes, _ = self.graph_transformer(nodes, edges, mask=mask)
        
        # Use the first node (sentence node) for classification
        sentence_nodes = nodes[:, 0]
        
        # Classify
        logits = self.classifier(sentence_nodes).squeeze(-1)
        return logits


class SentenceEncoderSarcasmDetector(nn.Module):
    """A simpler model that uses only a sentence encoder without graph transformer."""
    def __init__(self, sentence_encoder='sentence-transformers/all-MiniLM-L6-v2', device='cpu'):
        super().__init__()
        
        self.device = device
        
        # Load pre-trained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(sentence_encoder)
        self.encoder = AutoModel.from_pretrained(sentence_encoder)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, texts):
        # Tokenize texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                              max_length=128, return_tensors="pt").to(self.device)
        
        # Get model outputs
        outputs = self.encoder(**inputs)
        
        # Use CLS token embedding as sentence representation
        # sentence_emb = outputs.last_hidden_state[:, 0, :]

        token_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()  # [batch_size, seq_len, hidden_size]
        masked_embeddings = token_embeddings * attention_mask
        sum_embeddings = masked_embeddings.sum(1)
        sum_mask = attention_mask.sum(1)
        sentence_emb = sum_embeddings / sum_mask
        
        # Classify
        logits = self.classifier(sentence_emb).squeeze(-1)
        return logits