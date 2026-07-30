import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm.providers.ollama.ollama import OllamaProvider
from any_llm.types.completion import ChatCompletionMessage, CompletionParams


TOOL_CALLS: list[dict[str, Any]] = [
    {
        "id": "call_lookup",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
    }
]


@pytest.mark.asyncio
async def test_typed_assistant_tool_call_allows_none_content() -> None:
    """Typed OpenAI tool-call messages may omit content after model serialization."""
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=TOOL_CALLS,
    )

    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=Mock())

        with patch.object(OllamaProvider, "_convert_completion_response", return_value=Mock()):
            await provider.acompletion(model="llama3.1", messages=[message])

        sent_message = provider.client.chat.call_args.kwargs["messages"][0]
        assert sent_message == {
            "role": "assistant",
            "content": json.dumps(TOOL_CALLS),
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": None, "tool_calls": TOOL_CALLS},
        {"role": "assistant", "tool_calls": TOOL_CALLS},
    ],
)
async def test_raw_assistant_tool_call_allows_empty_content(message: dict[str, Any]) -> None:
    """Raw tool-call messages should not require text content."""
    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=Mock())

        with patch.object(OllamaProvider, "_convert_completion_response", return_value=Mock()):
            await provider._acompletion(CompletionParams(model_id="llama3.1", messages=[message]))

        sent_message = provider.client.chat.call_args.kwargs["messages"][0]
        assert sent_message == {
            "role": "assistant",
            "content": json.dumps(TOOL_CALLS),
        }


@pytest.mark.asyncio
async def test_assistant_tool_call_preserves_existing_content() -> None:
    """Existing assistant text should remain separated from serialized tool calls."""
    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=Mock())

        with patch.object(OllamaProvider, "_convert_completion_response", return_value=Mock()):
            await provider._acompletion(
                CompletionParams(
                    model_id="llama3.1",
                    messages=[
                        {
                            "role": "assistant",
                            "content": "Calling lookup",
                            "tool_calls": TOOL_CALLS,
                        }
                    ],
                )
            )

        sent_message = provider.client.chat.call_args.kwargs["messages"][0]
        assert sent_message == {
            "role": "assistant",
            "content": f"Calling lookup\n{json.dumps(TOOL_CALLS)}",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_calls", [None, []])
async def test_empty_tool_calls_are_not_serialized(tool_calls: list[Any] | None) -> None:
    """Absent tool calls should keep the assistant message unchanged."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
    }

    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=Mock())

        with patch.object(OllamaProvider, "_convert_completion_response", return_value=Mock()):
            await provider._acompletion(CompletionParams(model_id="llama3.1", messages=[message]))

        sent_message = provider.client.chat.call_args.kwargs["messages"][0]
        assert sent_message == message
