"""
train_sts_nli_mizan.py

Full training script for MizanEmbeddingModel-v1 using:
- STS-B (semantic similarity)
- SNLI (entailment/contradiction)
- MNLI (entailment/contradiction)

Requires:
    pip install datasets transformers torch mizanvector mizan-embedder
"""

import random
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from mizan_embedder.model import MizanEmbeddingModel
from mizan_embedder.data import TextPairDataset, make_collate_fn
from mizanvector.losses import MizanContrastiveLoss


# -------------------------------------------------------------
# STEP 1 — Load datasets
# -------------------------------------------------------------
def load_sts_pairs():
    print("Loading STS dataset...")
    sts = load_dataset("stsb_multi_mt", "en")

    pos = []
    neg = []

    for row in sts["train"]:
        s1 = row["sentence1"]
        s2 = row["sentence2"]
        sim = row["similarity_score"]

        if sim > 4.0:
            pos.append((s1, s2, 1.0))
        elif sim < 1.0:
            neg.append((s1, s2, 0.0))

    print(f"STS → positive: {len(pos)}, negative: {len(neg)}")
    return pos, neg


def load_nli_pairs():
    print("Loading SNLI dataset...")
    snli = load_dataset("snli")

    pos = []
    neg = []

    for row in snli["train"]:
        if row["label"] == 0:      # entailment
            pos.append((row["premise"], row["hypothesis"], 1.0))
        elif row["label"] == 2:    # contradiction
            neg.append((row["premise"], row["hypothesis"], 0.0))

    print("Loading MNLI dataset...")
    mnli = load_dataset("multi_nli")

    for row in mnli["train"]:
        if row["label"] == 0:
            pos.append((row["premise"], row["hypothesis"], 1.0))
        elif row["label"] == 2:
            neg.append((row["premise"], row["hypothesis"], 0.0))

    print(f"NLI → positive: {len(pos)}, negative: {len(neg)}")
    return pos, neg


# -------------------------------------------------------------
# STEP 2 — Build training dataset
# -------------------------------------------------------------
def build_training_pairs():
    sts_pos, sts_neg = load_sts_pairs()
    nli_pos, nli_neg = load_nli_pairs()

    pairs = sts_pos + sts_neg + nli_pos + nli_neg
    random.shuffle(pairs)

    print(f"TOTAL TRAINING PAIRS: {len(pairs)}")
    return pairs


# -------------------------------------------------------------
# STEP 3 — Train loop 32
# -------------------------------------------------------------
def train_mizan_encoder(
    backbone="distilbert-base-uncased",
    emb_dim=384,
    batch_size=32,
    epochs=3,
    lr=2e-5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Load pairs
    pairs = build_training_pairs()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    pairs = random.sample(pairs, 50000)
    # Dataset + DataLoader
    dataset = TextPairDataset(pairs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
    )

    # Model
    print("Initializing MizanEmbeddingModel...")
    model = MizanEmbeddingModel(
        backbone_name=backbone,
        emb_dim=emb_dim,
        pooling="mean",
        normalize=False,
    ).to(device)

    # Mizan Loss
    criterion = MizanContrastiveLoss(margin=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for enc1, enc2, labels in loader:
            enc1 = {k: v.to(device) for k, v in enc1.items()}
            enc2 = {k: v.to(device) for k, v in enc2.items()}
            labels = labels.to(device)

            emb1 = model(**enc1)
            emb2 = model(**enc2)

            loss = criterion(emb1, emb2, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(loader):.4f}")

    # Save final model
    save_dir = "MizanTextEncoder-base-384"
    print(f"Saving model to: {save_dir}")

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    torch.save(model.state_dict(), f"{save_dir}/pytorch_model.bin")
    print("Model saved successfully!")


# -------------------------------------------------------------
# RUN TRAINING
# -------------------------------------------------------------
if __name__ == "__main__":
    train_mizan_encoder(batch_size=4)
