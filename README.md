# Own Training Loop

A custom PyTorch training loop for fine-tuning Qwen 2.5 0.5B without HuggingFace Trainer.

## Live Demo
Not deployed—this is a learning project focused on understanding training mechanics.

## What It Does
- Loads Qwen 2.5 0.5B
- Tokenizes 80 Q&A pairs for Nimbus Coffee
- Trains using a custom PyTorch training loop
- No HuggingFace Trainer—all training logic written from scratch

## How It Works

1. **Load model** — Qwen 2.5 0.5B from HuggingFace
2. **Load dataset** — 80 Q&A pairs in JSONL format
3. **Tokenize** — Format as "### Instruction:\n...\n\n### Response:\n..."
4. **Initialize optimizer** — AdamW with learning rate 2e-5
5. **Training loop** — manual epochs, batches, forward pass, backward pass, weight updates
6. **Save model** — Model and tokenizer saved locally

## Training Results

### Run 1: Batch size 8, lr 5e-5, 5 epochs
- Epoch 1: loss 1.792
- Epoch 5: loss 0.037
- Result: Severe overfitting. Model memorized training data.

### Run 2: Batch size 8, lr 5e-5, 3 epochs
- Epoch 1: loss 1.792
- Epoch 3: loss 0.062
- Result: Still overfit. Generic answers on unseen questions.

### Run 3: Batch size 1 + gradient accumulation, lr 5e-5, 3 epochs
- Epoch 1: loss 1.792
- Epoch 3: loss 0.062
- Result: Better but still generic. Loss dropping too fast.

### Run 4: Batch size 1 + gradient accumulation, lr 2e-5, 3 epochs
- Epoch 1: loss 1.931
- Epoch 3: loss 0.124
- Result: Better loss curve but model still gives generic answers.

## Key Finding

Custom training loop works but does not generalize as well as HuggingFace Trainer. Trainer handles learning rate scheduling, weight decay, and gradient clipping automatically—these details matter for generalization.

## What I Learned

1. **Forward pass** — model(input_ids, attention_mask, labels) returns loss
2. **Backward pass** — loss.backward() computes gradients
3. **Weight update** — optimizer.step() applies gradients
4. **Gradient accumulation** — divide loss by steps, update every N examples
5. **Learning rate matters** — too high causes overfitting on small datasets
6. **Loss is not everything** — lower loss can mean memorization, not learning
7. **Trainer exists for a reason** — it handles details that are easy to miss

## Comparison with HuggingFace Trainer

| Method | Final Loss | Generalization |
|--------|-----------|----------------|
| Trainer (lr 5e-5, 5 epochs) | 0.193 | Moderate |
| Own loop (lr 5e-5, 5 epochs) | 0.037 | Poor |
| Own loop (lr 2e-5, 3 epochs) | 0.124 | Poor |

Trainer generalizes better despite higher loss because it uses learning rate scheduling and weight decay.

## Tech Stack

- Python
- PyTorch
- Transformers
- Datasets

## Project Structure

own-training-loop/
├── train.py            # Custom training loop
├── test-model.py       # Local model testing
├── dataset.jsonl       # 80 Q&A pairs
├── requirements.txt    # Dependencies
└── .gitignore          # Protects model files and secrets

## Skills Demonstrated

- PyTorch training loops
- Forward and backward passes
- Gradient accumulation
- Optimizer configuration
- Learning rate tuning
- Overfitting diagnosis
- Honest documentation of limitations

## Author

Built as part of a 10-day AI engineering learning sprint. Project 11.