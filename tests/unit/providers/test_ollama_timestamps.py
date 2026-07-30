from unittest.mock import Mock

import pytest
from ollama import ChatResponse as OllamaChatResponse, Message as OllamaMessage

from any_llm.providers.ollama.utils import (
    _create_chat_completion_from_ollama_response,
    _create_openai_chunk_from_ollama_chunk,
    _parse_ollama_timestamp,
)

EXPECTED_TIMESTAMP = 1785427200


@pytest.mark.parametrize(
    "created_at",
    [
        "2026-07-30T16:00:00Z",
        "2026-07-30T16:00:00.123Z",
        "2026-07-30T16:00:00.123456789Z",
        "2026-07-30T16:00:00+00:00",
    ],
)
def test_parse_ollama_timestamp_accepts_rfc3339_variants(created_at: str) -> None:
    """Ollama timestamps may omit or vary the fractional-second component."""
    assert _parse_ollama_timestamp(created_at) == EXPECTED_TIMESTAMP


def test_streaming_chunk_accepts_timestamp_without_fraction() -> None:
    """Streaming conversion should accept the zero-nanosecond Go time format."""
    message = Mock(spec=OllamaMessage)
    message.content = "Hello"
    message.thinking = None
    message.tool_calls = None
    message.role = "assistant"

    chunk = Mock(spec=OllamaChatResponse)
    chunk.message = message
    chunk.created_at = "2026-07-30T16:00:00Z"
    chunk.prompt_eval_count = None
    chunk.eval_count = None
    chunk.model = "llama3.1"
    chunk.done_reason = None

    result = _create_openai_chunk_from_ollama_chunk(chunk)

    assert result.created == EXPECTED_TIMESTAMP
    assert result.choices[0].delta.content == "Hello"


def test_completion_accepts_timestamp_without_fraction() -> None:
    """Non-streaming conversion should accept the zero-nanosecond Go time format."""
    message = Mock(spec=OllamaMessage)
    message.content = "Hello"
    message.thinking = None
    message.tool_calls = None
    message.role = "assistant"

    response = Mock(spec=OllamaChatResponse)
    response.message = message
    response.created_at = "2026-07-30T16:00:00Z"
    response.prompt_eval_count = 2
    response.eval_count = 3
    response.model = "llama3.1"
    response.done_reason = "stop"

    result = _create_chat_completion_from_ollama_response(response)

    assert result.created == EXPECTED_TIMESTAMP
    assert result.choices[0].message.content == "Hello"
