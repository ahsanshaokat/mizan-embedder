# How to Train Your First Mizan Encoder on Real Data

This guide describes how to train a Mizan-based text encoder using:

- `MizanEmbeddingModel`
- `MizanContrastiveLoss` (from `mizanvector`)
- Real datasets like STS / NLI (via `datasets` library)

Steps:

1. Load datasets (e.g., STS, SNLI, MNLI).
2. Build (text1, text2, label) pairs.
3. Wrap them in `TextPairDataset`.
4. Use `make_collate_fn(tokenizer)` to create a DataLoader.
5. Train `MizanEmbeddingModel` with `MizanContrastiveLoss`.
6. Save the model and tokenizer.
7. Use `MizanTextEncoderWrapper` for inference.

(You can expand this file with full dataset code as needed.)
