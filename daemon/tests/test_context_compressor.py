"""Unit tests for pilot.agents.context_compressor.ContextCompressor."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from pilot.agents.context_compressor import ContextCompressor
from pilot.models.token_counter import TOKEN_BUDGET, count_tokens


# ---------------------------------------------------------------------------
# Stub and helpers
# ---------------------------------------------------------------------------


class StubModelRouter:
    def __init__(self, response: str = "summary") -> None:
        self.generate = AsyncMock(return_value=response)


def make_compressor(response: str = "LLM summary") -> tuple[ContextCompressor, StubModelRouter]:
    router = StubModelRouter(response=response)
    return ContextCompressor(router), router


def _step(n: int, tokens: int = 10) -> str:
    """Generate a fake step string with approximately *tokens* tokens of content."""
    return f"Step {n}: " + ("error detail " * tokens)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_no_compression_when_under_budget():
    """compress() returns joined steps without calling LLM when under budget."""
    compressor, router = make_compressor()
    steps = [_step(i, tokens=5) for i in range(3)]
    result = await compressor.compress(steps, budget=TOKEN_BUDGET)

    assert result == "\n".join(steps)
    router.generate.assert_not_called()


async def test_compression_triggered_when_over_budget():
    """compress() calls the LLM to summarise older steps when over budget."""
    compressor, router = make_compressor(response="condensed summary of old steps")
    big_steps = [_step(i, tokens=1200) for i in range(10)]

    result = await compressor.compress(big_steps, budget=TOKEN_BUDGET)

    router.generate.assert_called_once()
    assert "condensed summary of old steps" in result


async def test_recent_steps_preserved_verbatim():
    """The last 2 steps must appear verbatim in the output when compression runs."""
    compressor, router = make_compressor(response="old summary")
    big_steps = [_step(i, tokens=1200) for i in range(10)]

    result = await compressor.compress(big_steps, budget=TOKEN_BUDGET)

    assert big_steps[-1] in result
    assert big_steps[-2] in result


async def test_fallback_to_truncation_on_llm_timeout():
    """compress() does not raise and returns a non-empty string when LLM times out."""
    router = StubModelRouter()
    router.generate = AsyncMock(side_effect=asyncio.TimeoutError())
    compressor = ContextCompressor(router)
    big_steps = [_step(i, tokens=1200) for i in range(10)]

    result = await compressor.compress(big_steps, budget=TOKEN_BUDGET)

    assert isinstance(result, str)
    assert len(result) > 0


async def test_empty_steps_returns_empty_string():
    """compress() with an empty step list returns empty string without calling LLM."""
    compressor, router = make_compressor()
    result = await compressor.compress([], budget=TOKEN_BUDGET)

    assert result == ""
    router.generate.assert_not_called()
