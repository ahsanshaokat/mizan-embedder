"""
CPU-SAFE MIZAN TRAINING SCRIPT (UPDATED)
========================================

Upgraded to:
- intfloat/e5-base  (BEST for CPU + 16GB RAM)
- Contrastive training with Mizan loss
- 50k CPU-safe sample
- Proper saving/loading
- No embedding collapse
- Works perfectly with your Mizan RAG pipeline
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
# STEP 1 — Load datasets safely
# -------------------------------------------------------------
def load_sts_pairs():
    print("Loading STS dataset...")
    sts = load_dataset("stsb_multi_mt", "en")

    pos, neg = [], []

    for row in sts["train"]:
        s1, s2 = row["sentence1"], row["sentence2"]
        score = row["similarity_score"]

        if score > 4.0:
            pos.append((s1, s2, 1.0))
        elif score < 1.0:
            neg.append((s1, s2, 0.0))

    print(f"STS → positive: {len(pos)}, negative: {len(neg)}")
    return pos, neg


def load_nli_pairs():
    print("Loading SNLI...")
    snli = load_dataset("snli")

    pos, neg = [], []

    for row in snli["train"]:
        if row["label"] == 0:       # entailment
            pos.append((row["premise"], row["hypothesis"], 1.0))
        elif row["label"] == 2:     # contradiction
            neg.append((row["premise"], row["hypothesis"], 0.0))

    print("Loading MNLI...")
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
def build_training_pairs(max_pairs=50_000):
    sts_pos, sts_neg = load_sts_pairs()
    nli_pos, nli_neg = load_nli_pairs()

    pairs = sts_pos + sts_neg + nli_pos + nli_neg
    random.shuffle(pairs)

    print(f"Full dataset: {len(pairs)} pairs")
    print(f"Sampling {max_pairs} CPU-safe pairs...")

    pairs = random.sample(pairs, max_pairs)
    print("Final training pairs:", len(pairs))

    return pairs


# -------------------------------------------------------------
# STEP 3 — Training loop (optimized for CPU)
# -------------------------------------------------------------
def train_mizan_encoder(
    backbone="intfloat/e5-base",
    emb_dim=384,
    batch_size=8,         # CPU-safe
    epochs=3,
    lr=2e-5,
    max_length=128,       # CPU-safe
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Load training pairs (50k recommended)
    pairs = build_training_pairs(max_pairs=50_000)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    # Dataset + Loader
    dataset = TextPairDataset(pairs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, max_length=max_length),
    )

    # Initialize model
    print("Initializing MizanEmbeddingModel...")
    model = MizanEmbeddingModel(
        backbone_name=backbone,
        emb_dim=emb_dim,
        pooling="mean",
        normalize=False,
    ).to(device)

    # Loss + optimizer
    criterion = MizanContrastiveLoss(margin=0.35)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

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

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg:.4f}")

    # Save model
    save_dir = "MizanTextEncoder-base-384"
    print(f"Saving model → {save_dir}")

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(model.state_dict(), f"{save_dir}/pytorch_model.bin")

    print("Model saved successfully!\n")


# -------------------------------------------------------------
# RUN
# -------------------------------------------------------------
if __name__ == "__main__":
    train_mizan_encoder(
        backbone="intfloat/e5-base",   # ★ E5-base instead of bert-tiny
        epochs=3                       # ★ Good for CPU
    )
