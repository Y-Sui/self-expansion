# Modal Implementation (Archived)

This directory contains the original Modal-based implementation files that have been replaced with OpenRouter API integration.

## Archived Files

- `modal_vllm_container.py` - Modal app for serving vLLM models with GPU
- `modal_embeddings.py` - Modal app for text embeddings using TEI
- `download_llama.py` - Script to download models to Modal volume storage

## Why Archived?

The project has been migrated from self-hosted Modal infrastructure to using OpenRouter API for LLM inference and OpenAI API for embeddings. This simplifies setup and reduces infrastructure management overhead.

If you want to use the original Modal implementation:
1. Restore these files to the root directory
2. Install `modal==0.72.39` in requirements.txt
3. Follow the original setup instructions in the commit history

## Migration Summary

- **LLM Inference**: Modal vLLM → OpenRouter API
- **Embeddings**: Modal TEI → OpenAI Embeddings API
- **Configuration**: VLLM_BASE_URL/VLLM_TOKEN → OPENROUTER_API_KEY/OPENAI_API_KEY
