# Generic LLM Testing Framework

This framework uses LangChain for orchestration and PromptFoo for testing LLM-powered use cases with custom REST APIs. Supports multi-run accuracy for non-determinism, tolerance-based assertions, and semantic checks.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set environment variables in `.env`:
   ```
   CUSTOM_LLM_BASE_URL=http://your-llm-server:port/v1
   CUSTOM_LLM_API_KEY=your_key_if_needed
   CUSTOM_LLM_MODEL=your_model
   ```

3. Run tests:
   ```
   promptfoo eval
   ```

4. View results:
   ```
   promptfoo view
   ```

## Extending

- Add post-processors in `providers/langchain_provider.py`.
- Add tests in `promptfooconfig.yaml`.

See code comments for details.
