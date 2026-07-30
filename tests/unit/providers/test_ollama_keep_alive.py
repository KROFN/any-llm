from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm.providers.ollama.ollama import OllamaProvider
from any_llm.types.completion import CompletionParams


@pytest.mark.asyncio
@pytest.mark.parametrize("keep_alive", ["5m", 0.0])
async def test_completion_passes_keep_alive_top_level(keep_alive: str | float) -> None:
    """Ollama keep_alive should not be nested inside model options."""
    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=Mock())

        with patch.object(OllamaProvider, "_convert_completion_response", return_value=Mock()):
            await provider._acompletion(
                CompletionParams(
                    model_id="llama3.1",
                    messages=[{"role": "user", "content": "Hello"}],
                ),
                keep_alive=keep_alive,
            )

        call_kwargs = provider.client.chat.call_args.kwargs
        assert call_kwargs["keep_alive"] == keep_alive
        assert "keep_alive" not in call_kwargs["options"]


@pytest.mark.asyncio
@pytest.mark.parametrize("keep_alive", ["5m", 0.0])
async def test_streaming_completion_passes_keep_alive_top_level(keep_alive: str | float) -> None:
    """Streaming should route keep_alive to the same top-level argument."""

    async def empty_async_iter() -> AsyncIterator[None]:
        return
        yield

    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=empty_async_iter())

        result = await provider._acompletion(
            CompletionParams(
                model_id="llama3.1",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            ),
            keep_alive=keep_alive,
        )
        async for _ in result:  # type: ignore[union-attr]
            pass

        call_kwargs = provider.client.chat.call_args.kwargs
        assert call_kwargs["keep_alive"] == keep_alive
        assert "keep_alive" not in call_kwargs["options"]
