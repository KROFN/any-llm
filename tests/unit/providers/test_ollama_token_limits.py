from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm.providers.ollama.ollama import OllamaProvider
from any_llm.types.completion import CompletionParams


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "max_tokens",
        "max_completion_tokens",
        "native_num_predict",
        "expected_num_predict",
    ),
    [
        (128, None, None, 128),
        (None, 256, None, 256),
        (128, 256, None, 256),
        (128, 256, 512, 512),
        (None, None, None, None),
    ],
)
async def test_completion_normalizes_token_limits(
    max_tokens: int | None,
    max_completion_tokens: int | None,
    native_num_predict: int | None,
    expected_num_predict: int | None,
) -> None:
    """Generic token limits should use Ollama's native num_predict option."""
    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=Mock())

        provider_kwargs: dict[str, Any] = {}
        if native_num_predict is not None:
            provider_kwargs["num_predict"] = native_num_predict

        with patch.object(OllamaProvider, "_convert_completion_response", return_value=Mock()):
            await provider._acompletion(
                CompletionParams(
                    model_id="llama3.1",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=max_tokens,
                    max_completion_tokens=max_completion_tokens,
                ),
                **provider_kwargs,
            )

        options = provider.client.chat.call_args.kwargs["options"]
        assert options.get("num_predict") == expected_num_predict
        assert "max_tokens" not in options
        assert "max_completion_tokens" not in options


@pytest.mark.asyncio
async def test_streaming_completion_normalizes_max_tokens() -> None:
    """Streaming should use the same token-limit normalization."""

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
                max_tokens=128,
                stream=True,
            )
        )
        async for _ in result:  # type: ignore[union-attr]
            pass

        options = provider.client.chat.call_args.kwargs["options"]
        assert options["num_predict"] == 128
        assert "max_tokens" not in options
        assert "max_completion_tokens" not in options
