import json
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import httpx

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.llms.klusterai.chat.transformation import KlusterAIConfig
from litellm.llms.klusterai.common_utils import KlusterAIMixin
from litellm.llms.klusterai.cost_calculator import cost_per_token

class TestKlusterAI:
    def test_completion(self):
        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "chat-12345",
                "object": "chat.completion",
                "created": 1736136422,
                "model": "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "This is a test response.",
                            "tool_calls": []
                        },
                        "logprobs": None,
                        "finish_reason": "stop",
                        "stop_reason": None
                    }
                ],
                "usage": {
                    "prompt_tokens": 48,
                    "total_tokens": 57,
                    "completion_tokens": 9
                }
            }
            mock_post.return_value = mock_response

            # Test basic completion
            response = litellm.completion(
                model="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            assert response.choices[0].message.content == "This is a test response."
            assert mock_post.called

            # Test with optional parameters
            response = litellm.completion(
                model="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                temperature=0.7,
                max_tokens=100,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                stop=["Stop"],
            )

            # Verify that parameters were passed correctly
            call_args = mock_post.call_args[1]["json"]
            assert call_args.get("temperature") == 0.7
            assert call_args.get("max_tokens") == 100
            assert call_args.get("top_p") == 0.9
            assert call_args.get("frequency_penalty") == 0.1
            assert call_args.get("presence_penalty") == 0.1
            assert call_args.get("stop") == ["Stop"]

    def test_streaming_completion(self):
        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = [
                json.dumps({
                    "id": "chat-12345",
                    "object": "chat.completion.chunk",
                    "created": 1736136422,
                    "model": "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                }),
                json.dumps({
                    "id": "chat-12345",
                    "object": "chat.completion.chunk",
                    "created": 1736136422,
                    "model": "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
                    "choices": [{"index": 0, "delta": {"content": "This"}, "finish_reason": None}]
                }),
                json.dumps({
                    "id": "chat-12345",
                    "object": "chat.completion.chunk",
                    "created": 1736136422,
                    "model": "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
                    "choices": [{"index": 0, "delta": {"content": " is"}, "finish_reason": None}]
                }),
                json.dumps({
                    "id": "chat-12345",
                    "object": "chat.completion.chunk",
                    "created": 1736136422,
                    "model": "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                })
            ]
            mock_post.return_value = mock_response

            response_stream = litellm.completion(
                model="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                stream=True
            )

            # Check that stream parameter was passed correctly
            assert mock_post.call_args[1]["json"]["stream"] is True

            # We can't directly assert on the stream contents as it's an iterator
            # But we can verify the stream was returned
            assert response_stream is not None

    @pytest.mark.asyncio
    async def test_async_completion(self):
        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json = AsyncMock(return_value={
                "id": "chat-12345",
                "object": "chat.completion",
                "created": 1736136422,
                "model": "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "This is an async test response.",
                            "tool_calls": []
                        },
                        "logprobs": None,
                        "finish_reason": "stop",
                        "stop_reason": None
                    }
                ],
                "usage": {
                    "prompt_tokens": 48,
                    "total_tokens": 57,
                    "completion_tokens": 9
                }
            })
            mock_post.return_value = mock_response

            response = await litellm.acompletion(
                model="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            assert response.choices[0].message.content == "This is an async test response."
            assert mock_post.called

    def test_config_validation(self):
        config = KlusterAIConfig()

        # Test API key validation
        with pytest.raises(ValueError, match="Missing API key for kluster.ai"):
            config.validate_environment(
                api_key=None,
                headers={},
                model="test-model",
                messages=[],
                optional_params={},
                litellm_params={}
            )

        # Test headers are properly set
        headers = config.validate_environment(
            api_key="test-key",
            headers={},
            model="test-model",
            messages=[],
            optional_params={},
            litellm_params={}
        )

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

        # Test with custom headers
        custom_headers = {"Custom-Header": "Value"}
        headers = config.validate_environment(
            api_key="test-key",
            headers=custom_headers,
            model="test-model",
            messages=[],
            optional_params={},
            litellm_params={}
        )

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"
        assert headers["Custom-Header"] == "Value"

        # Test with environment variable (mock)
        with patch.dict(os.environ, {"KLUSTER_AI_API_KEY": "env-key"}):
            with patch("litellm.secret_managers.main.get_secret_str", return_value="env-key"):
                headers = config.validate_environment(
                    api_key=None,
                    headers={},
                    model="test-model",
                    messages=[],
                    optional_params={},
                    litellm_params={}
                )
                assert headers["Authorization"] == "Bearer env-key"

    def test_transform_request(self):
        config = KlusterAIConfig()
        model = "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo"
        messages = [{"role": "user", "content": "Hello!"}]

        # Test basic transformation
        request_data = config.transform_request(
            model=model,
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={}
        )

        assert request_data["model"] == "Meta-Llama-3.1-8B-Instruct-Turbo"  # Model prefix removed
        assert request_data["messages"] == messages

        # Test with optional parameters
        optional_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "stop": ["Stop"],
            "stream": True
        }

        request_data = config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )

        assert request_data["temperature"] == 0.7
        assert request_data["max_tokens"] == 100
        assert request_data["top_p"] == 0.9
        assert request_data["frequency_penalty"] == 0.1
        assert request_data["presence_penalty"] == 0.1
        assert request_data["stop"] == ["Stop"]
        assert request_data["stream"] is True

    def test_get_provider_info(self):
        config = KlusterAIConfig()

        # Test for Llama models (supports function calling)
        llama_model = "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo"
        provider_info = config.get_provider_info(llama_model)
        assert provider_info.supports_function_calling is True

        # Test for non-Llama models
        non_llama_model = "klusterai/other-model"
        provider_info = config.get_provider_info(non_llama_model)
        assert provider_info.supports_function_calling is False

    def test_cost_calculator(self):
        from litellm.types.utils import Usage

        # Test Llama model pricing
        model = "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo"
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        with patch("litellm.utils.get_model_info") as mock_get_model_info:
            mock_get_model_info.return_value = {
                "input_cost_per_token": 0.00018,
                "output_cost_per_token": 0.00018
            }

            prompt_cost, completion_cost = cost_per_token(model, usage)
            assert prompt_cost == 0.00018 * 100
            assert completion_cost == 0.00018 * 50

        # Test fallback to default pricing when model not found
        with patch("litellm.utils.get_model_info") as mock_get_model_info:
            # First call raises exception (model not found)
            # Second call returns default pricing
            mock_get_model_info.side_effect = [
                Exception("Model not found"),
                {
                    "input_cost_per_token": 0.0001,
                    "output_cost_per_token": 0.0001
                }
            ]

            prompt_cost, completion_cost = cost_per_token("klusterai/unknown-model", usage)
            assert prompt_cost == 0.0001 * 100
            assert completion_cost == 0.0001 * 50

    def test_common_utils(self):
        mixin = KlusterAIMixin()

        # Test get_error_class
        error_class = mixin.get_error_class("Test error", 400, {})
        assert error_class.status_code == 400
        assert error_class.message == "Test error"

        # Test _get_api_key
        with patch("litellm.secret_managers.main.get_secret_str", return_value="env-key"):
            api_key = mixin._get_api_key(None)
            assert api_key == "env-key"

            api_key = mixin._get_api_key("explicit-key")
            assert api_key == "explicit-key"

        # Test validate_environment
        with patch("litellm.secret_managers.main.get_secret_str", return_value=None):
            with pytest.raises(ValueError, match="KLUSTER_AI_API_KEY is not set"):
                mixin.validate_environment(
                    headers={},
                    model="test-model",
                    messages=[],
                    optional_params={},
                    litellm_params={}
                )

        with patch("litellm.secret_managers.main.get_secret_str", return_value="env-key"):
            headers = mixin.validate_environment(
                headers={"Custom-Header": "Value"},
                model="test-model",
                messages=[],
                optional_params={},
                litellm_params={}
            )

            assert headers["Authorization"] == "Bearer env-key"
            assert headers["Custom-Header"] == "Value"
