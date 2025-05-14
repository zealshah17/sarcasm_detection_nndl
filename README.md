# Sarcasm Detection

This repository explores sarcasm detection using two complementary deep learning approaches designed for low-resource settings:

- **Graph Transformers**: Capture semantic and structural dependencies between tokens.
- **SetFit**: A prompt-free, few-shot learner that fine-tunes sentence embeddings using contrastive learning.

Both methods are evaluated on the **News Headlines** dataset and benchmarked against traditional baselines. We conduct detailed sensitivity analysis and ablation studies to assess performance trade-offs between structural modeling and data efficiency.

---

## Project Overview

We explore two contrasting deep learning methods under few-shot constraints:

- **Graph Transformer**: Uses pretrained sentence embeddings as node features and encodes structural relations between words. It models token-level interactions using edge-aware attention.
- **SetFit**: Adapts pretrained sentence encoders like RoBERTa, MiniLM, and MPNet using contrastive learning and a lightweight classifier, avoiding full transformer fine-tuning.

We benchmark both methods against:
- **TF-IDF + Logistic Regression**
- **Zero-shot classification** with `facebook/bart-large-mnli`

Both models significantly outperform the baselines on sarcasm detection tasks.

---

## How to Run the Graph Transformer

1. Navigate to the `graph/` directory.
2. Install dependencies:

```bash
pip install -r requirements.txt
```
## How to Run the Models

### Run the Graph Transformer Model

```bash
python train.py \
    --data_dir ./data \
    --model_type graph \
    --output_dir ./output_graph
```

### Run the Simple Sentence Encoder Baseline

```bash
python train.py \
    --data_dir ./data \
    --model_type simple \
    --output_dir ./output_simple
```

## Baseline Models

Implemented in the `baseline_models/baseline.ipynb` notebook:

- **TF-IDF + Logistic Regression**
- **Zero-shot Classification** using BART (`facebook/bart-large-mnli`)

---

## SetFit Experiments

Located in the `setfit_models/` directory, each notebook fine-tunes a different sentence transformer using the SetFit pipeline:

- `all-MiniLM-L6-v2`
- `all-mpnet-base-v2`
- `all-roberta-large-v1`

Training follows two stages:

1. **Contrastive learning** on sentence pairs.
2. **Classification head fine-tuning** on a few labeled examples.

---

## Requirements

Install all dependencies using:

```bash
pip install -r graph/requirements.txt
```

## Project Highlights

- Introduces **Graph Transformers** for sarcasm detection using node-edge encoding.
- Implements **SetFit** for few-shot sarcasm detection, evaluated with RoBERTa, MiniLM, and MPNet.
- Demonstrates that both models **outperform traditional baselines** under low supervision.
- Code is **modular**, **reproducible**, and **extensible** for other NLP classification tasks.


## Collaborators

- **Jasna Budhathoki**
- **Pranitha Natarajen**
- **Zeal Shah**



