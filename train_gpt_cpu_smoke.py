#!/usr/bin/env python3
"""
Minimal CPU-only smoke test for the local fineweb subset.

This keeps the same shard format and tokenizer checks as train_gpt_mlx.py,
but swaps in a tiny PyTorch model and short run defaults so it can execute
end-to-end on a Windows x64 CPU box without MLX.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None

    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")

    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(self, pattern: str):
        self.stream = TokenStream(pattern)

    def next_batch(self, batch_tokens: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = torch.from_numpy(chunk[:-1].reshape(-1, seq_len)).to(device=device, dtype=torch.long)
        y = torch.from_numpy(chunk[1:].reshape(-1, seq_len)).to(device=device, dtype=torch.long)
        return x, y


def load_validation_tokens(pattern: str, seq_len: int, max_tokens: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(file) for file in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if max_tokens > 0:
        usable = min(usable, (max_tokens // seq_len) * seq_len)
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden = dim * mlp_mult
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        attn_mask = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool).triu(1)
        h = self.ln1(x)
        y, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, dim: int, num_layers: int, num_heads: int, mlp_mult: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_mult) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = input_ids.size(1)
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = x @ self.tok_emb.weight.T
        if targets is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="mean")


def evaluate(model: TinyGPT, tokens: np.ndarray, seq_len: int, batch_tokens: int, device: torch.device) -> float:
    batch_tokens = max((batch_tokens // seq_len) * seq_len, seq_len)
    batch_seqs = batch_tokens // seq_len
    total_seqs = (tokens.size - 1) // seq_len
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.inference_mode():
        for seq_start in range(0, total_seqs, batch_seqs):
            seq_end = min(seq_start + batch_seqs, total_seqs)
            raw_start = seq_start * seq_len
            raw_end = seq_end * seq_len + 1
            chunk = tokens[raw_start:raw_end]
            x = torch.from_numpy(chunk[:-1].reshape(-1, seq_len)).to(device=device, dtype=torch.long)
            y = torch.from_numpy(chunk[1:].reshape(-1, seq_len)).to(device=device, dtype=torch.long)
            loss = model(x, y)
            token_count = y.numel()
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count
    model.train()
    return total_loss / max(total_tokens, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only local smoke test for parameter-golf.")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--out-dir", default="./logs")
    parser.add_argument("--run-id", default="cpu_smoke")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-tokens", type=int, default=256)
    parser.add_argument("--val-batch-tokens", type=int, default=512)
    parser.add_argument("--val-max-tokens", type=int, default=2048)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    checkpoint_path = out_dir / f"{args.run_id}.pt"

    def log(msg: str) -> None:
        print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"--tokenizer-path must point to a SentencePiece .model file: {args.tokenizer_path}")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    vocab_size = int(sp.vocab_size())
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )

    train_loader = TokenLoader(str(Path(args.data_path) / "fineweb_train_*.bin"))
    val_tokens = load_validation_tokens(
        str(Path(args.data_path) / "fineweb_val_*.bin"),
        args.seq_len,
        args.val_max_tokens,
    )

    model = TinyGPT(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_mult=args.mlp_mult,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log(f"run_id:{args.run_id}")
    log(f"device:{device}")
    log(f"dataset:{dataset_name}")
    if expected_train_files is None:
        log(f"train_shards:{actual_train_files}")
    else:
        log(f"train_shards:{actual_train_files}/{expected_train_files}")
    log(f"tokenizer_path:{args.tokenizer_path}")
    log(
        f"model:vocab_size={vocab_size} seq_len={args.seq_len} dim={args.model_dim} "
        f"layers={args.num_layers} heads={args.num_heads}"
    )
    log(
        f"iterations:{args.iterations} batch_tokens:{args.batch_tokens} "
        f"val_batch_tokens:{args.val_batch_tokens} val_max_tokens:{args.val_max_tokens}"
    )

    start_time = time.perf_counter()
    for step in range(1, args.iterations + 1):
        x, y = train_loader.next_batch(args.batch_tokens, args.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        elapsed_ms = 1000.0 * (time.perf_counter() - start_time)
        log(f"step:{step}/{args.iterations} train_loss:{loss.item():.4f} elapsed_ms:{elapsed_ms:.0f}")

    val_loss = evaluate(model, val_tokens, args.seq_len, args.val_batch_tokens, device)
    log(f"final_val_loss:{val_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "seq_len": args.seq_len,
                "model_dim": args.model_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "mlp_mult": args.mlp_mult,
            },
        },
        checkpoint_path,
    )
    log(f"saved_checkpoint:{checkpoint_path}")


if __name__ == "__main__":
    main()
