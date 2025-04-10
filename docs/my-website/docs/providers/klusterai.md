# klusterai

LiteLLM supports kluster.ai's chat completion models.

## Required API Keys

```python
import os
os.environ["KLUSTER_AI_API_KEY"] = "your-api-key"
```

## Usage

```python
import os
from litellm import completion

os.environ["KLUSTER_AI_API_KEY"] = "your-api-key"

# klusterai call
response = completion(
    model = "klusterai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

## Supported Models

| Model Name                               | Function Call                                                                        |
|-----------------------------------------|--------------------------------------------------------------------------------------|
| Llama 4 Maverick 17B 128E               | `completion('klusterai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8', messages)` |
| Llama 4 Scout 17B 16E                   | `completion('klusterai/meta-llama/Llama-4-Scout-17B-16E-Instruct', messages)`        |
| DeepSeek V3 0324                        | `completion('klusterai/deepseek-ai/DeepSeek-V3-0324', messages)`                     |
| DeepSeek R1                             | `completion('klusterai/deepseek-ai/DeepSeek-R1', messages)`                          |
| Gemma 3 27B                             | `completion('klusterai/google/gemma-3-27b-it', messages)`                            |
| Llama 3.1 8B                            | `completion('klusterai/Meta-Llama-3.1-8B-Instruct-Turbo', messages)`                 |
| Llama 3.1 405B                          | `completion('klusterai/Meta-Llama-3.1-405B-Instruct-Turbo', messages)`               |
| Llama 3.3 70B                           | `completion('klusterai/Meta-Llama-3.3-70B-Instruct-Turbo', messages)`                |
| Qwen2.5-VL-7B-Instruct                  | `completion('klusterai/Qwen/Qwen2.5-VL-7B-Instruct', messages)`                      |

## Asynchronous Inference

Kluster.AI supports asynchronous inference with a completion window option for cost management.

```python
from litellm import completion
import os

os.environ["KLUSTER_AI_API_KEY"] = "your-api-key"

# Async inference with 24-hour completion window
response = completion(
    model="klusterai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    messages=[{"role": "user", "content": "Analyze the investment potential of AI stocks"}],
    metadata={
        "@kluster.ai": {
            "async": True,
            "completion_window": "24h"
        }
    }
)
```

## Batch Processing

Kluster.AI supports batch processing for high-volume requests.

```python
from openai import OpenAI
import os

# Configure client with kluster.ai base URL
client = OpenAI(
    base_url="https://api.kluster.ai/v1",
    api_key=os.environ.get("KLUSTER_AI_API_KEY")
)

# Upload batch file
batch_file = client.files.create(
    file=open("batch_requests.jsonl", "rb"),
    purpose="batch"
)

# Create batch job
batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# Check batch status
batch_status = client.batches.retrieve(batch_job.id)
```

## Vision Support

Some kluster.ai models support vision capabilities.

```python
from litellm import completion
import base64, os

os.environ["KLUSTER_AI_API_KEY"] = "your-api-key"

# Encode image to base64
with open("image.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# Vision request
response = completion(
    model="klusterai/Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)
```
