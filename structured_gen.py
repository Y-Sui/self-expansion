from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict

import os
import dotenv

dotenv.load_dotenv()

# OpenRouter client for LLM
CLIENT = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# OpenAI client for embeddings (OpenRouter doesn't support embeddings)
EMBEDDINGS_CLIENT = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

print("Using base URL:", CLIENT.base_url)

# Use a default model or from environment variable
# OpenRouter supports many models - see https://openrouter.ai/models
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")

print("Using model:", DEFAULT_MODEL)

MAX_TOKENS = 12000


def messages(user: str, system: str = "You are a helpful assistant."):
    ms = [{"role": "user", "content": user}]
    if system:
        ms.insert(0, {"role": "system", "content": system})
    return ms


def generate(
    messages: List[Dict[str, str]],
    response_format: BaseModel,
) -> BaseModel:
    response = CLIENT.beta.chat.completions.parse(
        model=DEFAULT_MODEL,
        messages=messages,
        response_format=response_format,
        extra_body={
            # 'guided_decoding_backend': 'outlines',
            "max_tokens": MAX_TOKENS,
        },
    )
    return response


def generate_by_schema(
    messages: List[Dict[str, str]],
    schema: str,
) -> BaseModel:
    response = CLIENT.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        extra_body={
            # 'guided_decoding_backend': 'outlines',
            "max_tokens": MAX_TOKENS,
            "guided_json": schema,
        },
    )
    return response


def choose(
    messages: List[Dict[str, str]],
    choices: List[str],
) -> BaseModel:
    completion = CLIENT.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        extra_body={"guided_choice": choices, "max_tokens": MAX_TOKENS},
    )
    return completion.choices[0].message.content


def regex(
    messages: List[Dict[str, str]],
    regex: str,
) -> BaseModel:
    completion = CLIENT.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        extra_body={"guided_regex": regex, "max_tokens": MAX_TOKENS},
    )
    return completion.choices[0].message.content


def embed(content: str) -> List[float]:
    """Generate embeddings using OpenAI's embedding API."""
    response = EMBEDDINGS_CLIENT.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        input=content,
    )
    return response.data[0].embedding
