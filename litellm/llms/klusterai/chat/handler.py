"""
Handler for kluster.ai chat completions endpoint.
"""

from typing import Callable, Dict, List, Optional, Union

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.utils import ModelResponse

from .transformation import KlusterAIConfig


class KlusterAIChatCompletion:
    def __init__(self) -> None:
        pass

    async def acompletion_stream_function(
        self,
        model: str,
        messages: List,
        api_base: str,
        custom_prompt_dict: Dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Union[float, httpx.Timeout],
        client: Optional[AsyncHTTPHandler],
        encoding,
        api_key,
        logging_obj,
        stream,
        _is_function_call,
        data: Dict,
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
    ):
        # Make sure stream is set to True in data
        data["stream"] = True

        try:
            if client is None:
                client = get_async_httpx_client(
                    llm_provider=litellm.LlmProviders.KLUSTER_AI
                )

            response = await client.post(
                api_base, headers=headers, json=data, timeout=timeout, stream=True
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            response_content = await e.response.aread()
            response_text = response_content.decode('utf-8', errors='replace') if isinstance(response_content, bytes) else str(response_content)
            raise Exception(f"HTTP error occurred: {e} - {response_text}")
        except Exception as e:
            raise Exception(f"Error occurred: {e}")

        from litellm.utils import CustomStreamWrapper

        streamwrapper = CustomStreamWrapper(
            completion_stream=response.aiter_lines(),
            model=model,
            custom_llm_provider="klusterai",
            logging_obj=logging_obj,
        )

        return streamwrapper

    async def acompletion_function(
        self,
        model: str,
        messages: List,
        api_base: str,
        custom_prompt_dict: Dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Union[float, httpx.Timeout],
        encoding,
        api_key,
        logging_obj,
        stream,
        _is_function_call,
        data: Dict,
        optional_params: Dict,
        json_mode: bool,
        litellm_params: Dict,
        provider_config: BaseConfig,
        logger_fn=None,
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ModelResponse:
        try:
            if client is None:
                client = get_async_httpx_client(
                    llm_provider=litellm.LlmProviders.KLUSTER_AI
                )

            response = await client.post(
                api_base, headers=headers, json=data, timeout=timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            response_content = await e.response.aread()
            response_text = response_content.decode('utf-8', errors='replace') if isinstance(response_content, bytes) else str(response_content)
            raise Exception(f"HTTP error occurred: {e} - {response_text}")
        except Exception as e:
            raise Exception(f"Error occurred: {e}")

        return provider_config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            json_mode=json_mode,
        )

    def completion(
        self,
        model: str,
        messages: List,
        api_base: str,
        custom_llm_provider: str,
        custom_prompt_dict: Dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: Dict,
        timeout: Union[float, httpx.Timeout],
        litellm_params: Dict,
        acompletion=None,
        logger_fn=None,
        headers={},
        client=None,
    ):
        stream = optional_params.pop("stream", False)
        json_mode = optional_params.pop("json_mode", False)
        _is_function_call = False

        provider_config = KlusterAIConfig()
        headers = provider_config.validate_environment(
            api_key=api_key,
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        data = provider_config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Log the request
        logging_obj.pre_call(
            input=messages,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        if acompletion is True:
            if stream is True:
                return self.acompletion_stream_function(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    timeout=timeout,
                    client=client,
                    encoding=encoding,
                    api_key=api_key,
                    logging_obj=logging_obj,
                    stream=stream,
                    _is_function_call=_is_function_call,
                    data=data,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                )
            else:
                return self.acompletion_function(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    timeout=timeout,
                    encoding=encoding,
                    api_key=api_key,
                    logging_obj=logging_obj,
                    stream=stream,
                    _is_function_call=_is_function_call,
                    data=data,
                    optional_params=optional_params,
                    json_mode=json_mode,
                    litellm_params=litellm_params,
                    provider_config=provider_config,
                    logger_fn=logger_fn,
                    headers=headers,
                    client=client,
                )
        else:
            if stream is True:
                if client is None:
                    client = HTTPHandler(timeout=timeout)

                data["stream"] = True
                response = client.post(
                    api_base, headers=headers, json=data, timeout=timeout, stream=True
                )
                response.raise_for_status()

                from litellm.utils import CustomStreamWrapper

                streamwrapper = CustomStreamWrapper(
                    completion_stream=response.iter_lines(),
                    model=model,
                    custom_llm_provider="klusterai",
                    logging_obj=logging_obj,
                )

                return streamwrapper
            else:
                if client is None:
                    client = HTTPHandler(timeout=timeout)

                response = client.post(
                    api_base, headers=headers, json=data, timeout=timeout
                )
                response.raise_for_status()

                return provider_config.transform_response(
                    model=model,
                    raw_response=response,
                    model_response=model_response,
                    logging_obj=logging_obj,
                    api_key=api_key,
                    request_data=data,
                    messages=messages,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    encoding=encoding,
                    json_mode=json_mode,
                )
