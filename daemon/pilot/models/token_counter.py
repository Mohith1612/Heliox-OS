"""Lightweight token counting with optional tiktoken acceleration."""
from __future__ import annotations

TOKEN_BUDGET = 8000


def count_tokens(text: str) -> int:
    """Estimate token count for *text*.

    Uses tiktoken cl100k_base encoding when available; falls back to the
    char-count heuristic ``len(text) // 4`` so that tiktoken is never a
    hard dependency.
    """
    try:
        import tiktoken  # type: ignore[import]

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4
