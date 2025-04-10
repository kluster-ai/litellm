"""
For calculating cost of Kluster AI serverless inference models.
"""

from typing import Tuple

from litellm.types.utils import Usage
from litellm.utils import get_model_info


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing token usage information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    # Get model pricing information
    try:
        model_info = get_model_info(model=model, custom_llm_provider="klusterai")
    except Exception:
        # If model not found in pricing data, default to a generic pricing structure
        model_info = get_model_info(model="kluster-ai-default", custom_llm_provider="klusterai")

    # Calculate input cost
    prompt_cost: float = usage["prompt_tokens"] * model_info["input_cost_per_token"]

    # Calculate output cost
    completion_cost = usage["completion_tokens"] * model_info["output_cost_per_token"]

    return prompt_cost, completion_cost
