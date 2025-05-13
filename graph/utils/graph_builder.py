import torch
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import networkx as nx

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def build_token_graph(text, tokenizer=None):
    """
    Build a simple graph from tokens in the text.
    
    Args:
        text: Input text
        tokenizer: Optional tokenizer function
        
    Returns:
        nodes: List of tokens
        edges: Adjacency matrix
    """
    # Tokenize text
    if tokenizer:
        tokens = tokenizer(text)
    else:
        tokens = word_tokenize(text)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes (tokens)
    for i, token in enumerate(tokens):
        G.add_node(i, token=token)
    
    # Add edges (connections between tokens)
    # Here we connect adjacent tokens for simplicity
    for i in range(len(tokens) - 1):
        G.add_edge(i, i + 1)
        G.add_edge(i + 1, i)  # Bidirectional
    
    # Get adjacency matrix
    adj_matrix = nx.to_numpy_array(G)
    
    return tokens, adj_matrix


def build_sentence_graph(text):
    """
    Build a graph from sentences in the text.
    
    Args:
        text: Input text
        
    Returns:
        nodes: List of sentences
        edges: Adjacency matrix
    """
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Create a fully connected graph
    G = nx.complete_graph(len(sentences))
    
    # Get adjacency matrix
    adj_matrix = nx.to_numpy_array(G)
    
    return sentences, adj_matrix


def text_to_dependency_graph(text, nlp=None):
    """
    Convert text to a dependency graph.
    Requires spaCy to be installed and a language model to be loaded.
    
    Args:
        text: Input text
        nlp: spaCy NLP model
        
    Returns:
        nodes: List of tokens
        edges: Adjacency matrix
    """
    # If spaCy is not available, fall back to simple token graph
    try:
        import spacy
        if nlp is None:
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy model not found. Using simple token graph instead.")
                return build_token_graph(text)
    except ImportError:
        print("spaCy not installed. Using simple token graph instead.")
        return build_token_graph(text)
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes (tokens)
    for i, token in enumerate(doc):
        G.add_node(i, token=token.text, pos=token.pos_)
    
    # Add edges (dependencies)
    for token in doc:
        if token.head.i != token.i:  # Not the root
            G.add_edge(token.i, token.head.i, dep=token.dep_)
    
    # Get adjacency matrix
    adj_matrix = nx.to_numpy_array(G)
    
    return [token.text for token in doc], adj_matrix


def prepare_graph_features(texts, tokenizer=None, max_nodes=128):
    """
    Prepare graph features for a batch of texts.
    
    Args:
        texts: List of input texts
        tokenizer: Optional tokenizer function
        max_nodes: Maximum number of nodes in the graph
        
    Returns:
        nodes_batch: Tensor of node features (batch_size, max_nodes, node_dim)
        adj_matrices_batch: Tensor of adjacency matrices (batch_size, max_nodes, max_nodes)
        masks_batch: Tensor of node masks (batch_size, max_nodes)
    """
    batch_size = len(texts)
    nodes_batch = []
    adj_matrices_batch = []
    masks_batch = []
    
    for text in texts:
        # Build graph
        nodes, adj_matrix = build_token_graph(text, tokenizer)
        
        # Create mask
        mask = torch.zeros(max_nodes, dtype=torch.bool)
        mask[:len(nodes)] = True
        
        # Pad adjacency matrix
        padded_adj = np.zeros((max_nodes, max_nodes))
        padded_adj[:len(nodes), :len(nodes)] = adj_matrix
        
        # For now, we use a simple one-hot encoding for nodes
        # In a real implementation, you would replace this with proper node embeddings
        node_features = np.zeros((max_nodes, 1))
        node_features[:len(nodes), 0] = 1.0
        
        nodes_batch.append(node_features)
        adj_matrices_batch.append(padded_adj)
        masks_batch.append(mask)
    
    # Convert to tensors
    nodes_tensor = torch.tensor(np.stack(nodes_batch), dtype=torch.float)
    adj_tensor = torch.tensor(np.stack(adj_matrices_batch), dtype=torch.float)
    masks_tensor = torch.stack(masks_batch)
    
    return nodes_tensor, adj_tensor, masks_tensor