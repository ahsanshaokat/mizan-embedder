import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Literal, Optional, Dict

PoolingType = Literal["mean", "cls", "max"]


# ============================================================================
#                           Mizan Embedding Model
# ============================================================================

class MizanEmbeddingModel(nn.Module):
    """
    Core embedding model for Mizan embeddings.
    Wraps a HuggingFace encoder + projection + pooling + normalization.
    """

    def __init__(
        self,
        backbone_name: str = "distilbert-base-uncased",
        emb_dim: int = 384,
        pooling: PoolingType = "mean",
        normalize: bool = False,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.backbone = AutoModel.from_pretrained(backbone_name)

        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, emb_dim)

        self.pooling = pooling
        self.normalize = normalize

    # ----------------------------------------------------------------------

    def _pool(self, token_embeddings, attention_mask):
        if self.pooling == "cls":
            return token_embeddings[:, 0, :]

        mask = attention_mask.unsqueeze(-1).float()

        if self.pooling == "mean":
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-8)
            return summed / counts

        if self.pooling == "max":
            replaced = token_embeddings.masked_fill(mask == 0, -1e9)
            return replaced.max(dim=1).values

        raise ValueError(f"Unknown pooling type: {self.pooling}")

    # ----------------------------------------------------------------------

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        token_embeddings = outputs.last_hidden_state
        pooled = self._pool(token_embeddings, attention_mask)
        emb = self.proj(pooled)

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)

        return emb

    # ----------------------------------------------------------------------
    # Save Model
    # ----------------------------------------------------------------------

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        config = {
            "backbone_name": self.backbone_name,
            "emb_dim": self.proj.out_features,
            "pooling": self.pooling,
            "normalize": self.normalize,
        }

        import json
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved MizanEmbeddingModel → {save_directory}")

    # ----------------------------------------------------------------------
    # Load Model
    # ----------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, folder, device="cpu"):
        import json

        config_path = os.path.join(folder, "config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)

        model = cls(
            backbone_name=cfg["backbone_name"],
            emb_dim=cfg["emb_dim"],
            pooling=cfg["pooling"],
            normalize=cfg["normalize"],
        )

        weights_path = os.path.join(folder, "pytorch_model.bin")
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)

        return model


# ============================================================================
#                       Mizan Text Encoder Wrapper (with Caching)
# ============================================================================

class MizanTextEncoderWrapper:
    """
    A universal encoder wrapper that supports:

    ✔ HuggingFace models
    ✔ Custom local Mizan models
    ✔ Optional caching for embeddings
    ✔ encode() and encode_one()
    """

    def __init__(
        self,
        backbone_name: str = "distilbert-base-uncased",
        emb_dim: int = 384,
        pooling: PoolingType = "mean",
        normalize: bool = False,
        device: Optional[str] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Parameters
        ----------
        backbone_name : str
            HuggingFace model OR folder containing custom Mizan model.

        cache : dict
            Optional. If provided, embeddings will be cached in memory.
        """

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = cache  # {"text": tensor}

        # ------------------------------------------
        # Load custom trained model (folder)
        # ------------------------------------------
        if os.path.isdir(backbone_name):
            print(f"Loading custom trained Mizan model from: {backbone_name}")

            self.model = MizanEmbeddingModel.from_pretrained(
                backbone_name, device=self.device
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model.backbone_name
            )

        else:
            # ------------------------------------------
            # Load HF model
            # ------------------------------------------
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)

            self.model = MizanEmbeddingModel(
                backbone_name=backbone_name,
                emb_dim=emb_dim,
                pooling=pooling,
                normalize=normalize,
            ).to(self.device)

        self.model.eval()

    # ============================================================================
    #                          Caching Helpers
    # ============================================================================

    def _maybe_from_cache(self, text: str):
        if self.cache is None:
            return None
        return self.cache.get(text)

    def _save_to_cache(self, text: str, emb: torch.Tensor):
        if self.cache is not None:
            self.cache[text] = emb

    # ============================================================================
    #                          Batch Encoder
    # ============================================================================

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 16) -> torch.Tensor:

        cached_outputs = {}
        to_encode = []

        # 1) Check cache
        for txt in texts:
            cached = self._maybe_from_cache(txt)
            if cached is not None:
                cached_outputs[txt] = cached
            else:
                to_encode.append(txt)

        # 2) Encode remaining texts
        if to_encode:
            for i in range(0, len(to_encode), batch_size):
                batch = to_encode[i:i + batch_size]

                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}

                emb = self.model(**enc).cpu()

                # store into cache
                for t, e in zip(batch, emb):
                    self._save_to_cache(t, e)

                # also store in output set
                for t, e in zip(batch, emb):
                    cached_outputs[t] = e

        # 3) Return embeddings in the same order as input
        ordered = [cached_outputs[txt] for txt in texts]
        return torch.stack(ordered)

    # ============================================================================
    #                          Single Encoder
    # ============================================================================

    @torch.no_grad()
    def encode_one(self, text: str) -> torch.Tensor:
        cached = self._maybe_from_cache(text)
        if cached is not None:
            return cached

        emb = self.encode([text])[0]
        return emb
