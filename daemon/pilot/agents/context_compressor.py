"""Rolling context-window compressor for the ReAct retry loop."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from pilot.models.token_counter import count_tokens

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.context_compressor")

_COMPRESSION_PROMPT = (
    "Summarize these past agent steps in under 200 words, keeping only facts "
    "relevant to diagnosing a failure. Do not include successful steps.\n\n"
    "Steps:\n{older_steps}"
)
_COMPRESSION_TIMEOUT_SECONDS = 7


class ContextCompressor:
    """Compresses a step log to fit within a token budget.

    When the concatenated steps exceed *budget* tokens the compressor keeps
    the last two steps verbatim and summarises all older steps via an LLM
    call (7-second timeout). Falls back to character-level tail truncation if
    the LLM call fails or times out.
    """

    def __init__(self, model_router: ModelRouter) -> None:
        self._router = model_router

    async def compress(self, steps: list[str], budget: int) -> str:
        """Return a compressed representation of *steps* that fits in *budget* tokens.

        Returns the steps joined as-is when already within budget (no LLM call).
        """
        if not steps:
            return ""

        joined = "\n".join(steps)
        if count_tokens(joined) <= budget:
            return joined

        recent = steps[-2:]
        older = steps[:-2]
        older_text = "\n".join(older)
        recent_text = "\n".join(recent)

        if not older_text:
            return self._char_truncate(joined, budget)

        summary = await self._summarise(older_text)
        compressed = f"{summary}\n\n[Recent steps — verbatim]\n{recent_text}"

        if count_tokens(compressed) > budget:
            compressed = self._char_truncate(compressed, budget)

        return compressed

    async def _summarise(self, older_steps: str) -> str:
        prompt = _COMPRESSION_PROMPT.format(older_steps=older_steps)
        try:
            async with asyncio.timeout(_COMPRESSION_TIMEOUT_SECONDS):
                return await self._router.generate(
                    prompt,
                    system="You are a concise technical summariser.",
                    temperature=0.0,
                )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            logger.warning("Context compression timed out (%s); falling back to truncation", exc)
            return self._char_truncate(older_steps, 500)
        except Exception as exc:
            logger.warning("Context compression failed (%s); falling back to truncation", exc)
            return self._char_truncate(older_steps, 500)

    @staticmethod
    def _char_truncate(text: str, token_budget: int) -> str:
        """Truncate *text* to approximately *token_budget* tokens (4 chars/token)."""
        char_limit = token_budget * 4
        if len(text) <= char_limit:
            return text
        return "...[truncated]\n" + text[-char_limit:]
