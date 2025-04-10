from typing import Any, List, Optional, Union

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ProviderSpecificModelInfo

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class KlusterAIConfig(OpenAIGPTConfig):
    """
    Reference: https://api.kluster.ai/v1/chat/completions API

    The class `KlusterAIConfig` provides configuration for kluster.ai's Chat Completions API interface.
    """

    max_tokens: Optional[int] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    frequency_penalty: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    seed: Optional[int] = None
    user: Optional[str] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        frequency_penalty: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str):
        return [
            "stream",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "logprobs",
            "top_logprobs",
            "seed",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)

        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                if value is not None:
                    optional_params[param] = value

        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def get_provider_info(self, model: str) -> ProviderSpecificModelInfo:
        # Update to indicate tool support for Llama models
        is_llama_model = "llama" in model.lower()

        provider_specific_model_info = ProviderSpecificModelInfo(
            supports_function_calling=is_llama_model,
            supports_prompt_caching=False,
            supports_pdf_input=False,
            supports_vision=False,
            supports_tool_choice=is_llama_model
        )
        return provider_specific_model_info

    def validate_environment(
        self,
        headers: dict[Any, Any],
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict[Any, Any],
        litellm_params: dict[Any, Any],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict[Any, Any]:
        # Get API key if not provided
        api_key = api_key or get_secret_str("KLUSTER_AI_API_KEY")
        if api_key is None:
            raise ValueError("Missing API key for kluster.ai")

        # Add authorization header
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"

        return headers
