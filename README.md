# Abstractive Text Summarization with BART

## Overview

This project fine-tunes a pre-trained BART (Bidirectional and Auto-Regressive Transformers) model for abstractive text summarization of multi-party dialogues. Rather than extracting key sentences, the model generates entirely new, compressed summaries that capture the essence of conversations.

The model is fine-tuned on dialogue data focused on academic conversations (graduate students discussing machine learning, business analytics, and grad school applications).

## Features

- **Pre-trained BART Model** — 140M parameter model from KerasHub (`bart_base_en`)
- **Transfer Learning** — Fine-tuned on domain-specific dialogue data
- **Seq2Seq Architecture** — Bidirectional encoder + autoregressive decoder
- **Flexible Backend** — Keras 3 supports TensorFlow, JAX, or PyTorch
- **Mixed Precision Training** — Memory-efficient training with automatic loss scaling

## Architecture

| Component | Details |
|-----------|---------|
| Model | BartSeq2SeqLM (bart_base_en) |
| Parameters | 139.4M |
| Tokenizer | SentencePiece (50,265 vocab) |
| Optimizer | AdamW with weight decay |
| Loss | Sparse Categorical Crossentropy |

## Dataset

This project uses dialogue-summary pairs in JSON format. You need to provide your own data files in the `data/` directory.

### Data Format

Each JSON file should contain an array of objects:

```json
[
  {
    "id": "30000001",
    "dialogue": "Alex: Have you started your application?\nSam: Yes, I'm working on it.",
    "summary": "Alex and Sam discuss grad school applications."
  }
]
```

### Setup

1. Create a `data/` directory in the project root:
   ```bash
   mkdir -p data
   ```

2. Place your data files as:
   ```
   data/train_new.json   # Training data
   data/val_new.json     # Validation data
   data/test_new.json    # Test data
   ```

3. **(Optional)** The notebook also supports the **SAMSum Corpus** (~15,000 dialogue-summary pairs). To use it:
   - Download from [SAMSum on HuggingFace](https://huggingface.co/datasets/samsum) or the [original paper](https://arxiv.org/abs/1911.12237)
   - Extract the corpus into `data/corpus/`:
     ```
     data/corpus/train.json
     data/corpus/val.json
     data/corpus/test.json
     ```
   - Note: SAMSum is licensed CC BY-NC-ND 4.0 (non-commercial use only)

## Requirements

- Python 3.9+
- Keras 3.x
- KerasHub
- TensorFlow 2.15+
- JAX (optional, for faster training)
- py7zr (if using compressed corpus)

## Installation

```bash
pip install keras keras-hub tensorflow tensorflow-datasets py7zr
# Optional: for JAX backend
pip install jax
```

## Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook abstractive_summarization_with_bart.ipynb
   ```

2. Run all cells sequentially. The notebook will:
   - Load and preprocess dialogue-summary pairs from `data/`
   - Load the pre-trained BART model
   - Fine-tune on your training data
   - Generate and evaluate summaries

### Configuration

Key parameters (adjustable in the notebook):

```python
BATCH_SIZE = 8
EPOCHS = 5
MAX_ENCODER_SEQUENCE_LENGTH = 64   # Max dialogue tokens
MAX_DECODER_SEQUENCE_LENGTH = 32   # Max summary tokens
MAX_GENERATION_LENGTH = 40         # Max tokens at inference
```

### Switching Backends

```bash
export KERAS_BACKEND=tensorflow  # default
export KERAS_BACKEND=jax         # faster for some hardware
export KERAS_BACKEND=torch       # PyTorch backend
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Batch Size | 8 |
| Learning Rate | 5e-5 |
| Weight Decay | 0.01 |
| Epochs | 5 |
| Encoder Max Length | 64 tokens |
| Decoder Max Length | 32 tokens |

## Results

The fine-tuned model:
- Identifies conversation topics and participants accurately
- Generates concise, grammatically correct summaries
- Works well with limited training data via transfer learning
- Can be improved with more data and longer training

## Project Structure

```
Text_Summarization/
├── abstractive_summarization_with_bart.ipynb   # Main notebook
├── README.md
└── data/                                        # You provide this
    ├── train_new.json
    ├── val_new.json
    ├── test_new.json
    └── corpus/                                  # Optional: SAMSum corpus
        ├── train.json
        ├── val.json
        └── test.json
```

## References

- [BART Paper](https://arxiv.org/abs/1910.13461) — Lewis et al., 2019
- [KerasHub](https://keras.io/keras_hub/)
- [SAMSum Dataset Paper](https://arxiv.org/abs/1911.12237) — Gliwa et al., 2019

## Author

Ben Laufer — GSB 545, Cal Poly SLO
