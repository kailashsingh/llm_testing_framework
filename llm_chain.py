import os
import json
import requests
from typing import Dict, Any, Optional, Callable
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader  # For documents
from dotenv import load_dotenv

load_dotenv()

class CustomLLM:
    """Custom REST-based LLM client."""
    def __init__(self, base_url: str, api_key: Optional[str] = None, model: str = "default", temperature: float = 0.1, timeout: int = 60):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def invoke(self, messages: list[Dict[str, str]]) -> str:
        """Call the LLM API with messages (OpenAI-style). Adjust for custom payload."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 1024,  # Configurable
        }
        
        # Endpoint: Adjust if not OpenAI-compatible (e.g., /generate)
        url = f"{self.base_url}/chat/completions"  # Or /generate for custom
        
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # Extract content: Adjust path for custom response format (e.g., data['result'])
            content = data["choices"][0]["message"]["content"]
            return content
        except requests.exceptions.RequestException as e:
            raise Exception(f"API call failed: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Invalid response format: {e}")

class GenericLLMProcessor:
    """Generic processor for LLM use cases: Prompt + Context -> LLM -> Optional Post-Processing."""
    def __init__(
        self, 
        base_url: str, 
        api_key: Optional[str] = None, 
        model: str = "default", 
        temperature: float = 0.1,
        post_processor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ):
        self.llm = CustomLLM(base_url, api_key, model, temperature)
        self.parser = StrOutputParser()
        self.post_processor = post_processor or self._default_post_processor
    
    def load_context(self, doc_path: str) -> str:
        """Load context from document (e.g., PDF report)."""
        loader = PyPDFLoader(doc_path)
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])  # Simple concat; use splitters for RAG if needed
    
    def create_prompt(self, prompt_template: str, context: str, user_query: str = "") -> list[Dict[str, str]]:
        """Convert to messages format for API (system + user)."""
        full_prompt = prompt_template + f"\n\nContext: {context}\n\nQuery: {user_query}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Output in the requested format (e.g., JSON if specified)."},
            {"role": "user", "content": full_prompt}
        ]
        return messages
    
    def invoke(self, prompt_template: str, context: str, user_query: Optional[str] = None, doc_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the chain: Prompt + Context -> Custom LLM -> Parsed Output -> Post-Processing."""
        if doc_path:
            context = self.load_context(doc_path)
        
        messages = self.create_prompt(prompt_template, context, user_query or "")
        response = self.llm.invoke(messages)
        
        try:
            # Assume output is JSON if parseable; otherwise, treat as raw text
            if response.strip().startswith("{"):
                extracted = json.loads(response)
            else:
                extracted = {"raw_output": response}
            derived = self.post_processor(extracted)  # Domain-specific hook
            return {"extracted": extracted, "derived": derived}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON output", "raw": response}
    
    def _default_post_processor(self, extracted: Dict[str, Any]) -> Dict[str, Any]:
        """No-op default; override for use-case specific derivations."""
        return {}

# Usage example (for manual testing)
if __name__ == "__main__":
    processor = GenericLLMProcessor(
        base_url=os.getenv("CUSTOM_LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("CUSTOM_LLM_API_KEY"),
        model=os.getenv("CUSTOM_LLM_MODEL", "default")
    )
    result = processor.invoke(
        prompt_template="Analyze sentiment as JSON: sentiment_score (0-1 float).",
        context="This product is amazing and exceeded expectations!",
        # Pass a post_processor=your_sentiment_func here if testing manually
    )
    print(json.dumps(result, indent=2))
