import sys
import json
import os
from llm_chain import GenericLLMProcessor  # Updated import

# Define post-processors here for modularity
def default_post_processor(extracted):
    return {}

def financial_post_processor(extracted):
    """Example: Derive profit margin."""
    if "revenue" in extracted and "profit" in extracted:
        margin = (extracted["profit"] / extracted["revenue"]) * 100 if extracted["revenue"] != 0 else 0
        return {"profit_margin": round(margin, 2)}
    return {}

def sentiment_post_processor(extracted):
    """Example: Map sentiment score to label (positive/negative/neutral)."""
    if "sentiment_score" in extracted:
        score = extracted["sentiment_score"]
        if score > 0.5:
            label = "positive"
        elif score < 0.5:
            label = "negative"
        else:
            label = "neutral"
        # Optional: Add confidence or other derivations
        return {"sentiment_label": label, "confidence": abs(score - 0.5) * 2}  # 0-1 confidence
    return {}

def run_prompt(prompt, vars):
    """PromptFoo provider: Takes prompt/vars, returns output."""
    # Override with vars or env
    base_url = vars.get("base_url", os.getenv("CUSTOM_LLM_BASE_URL"))
    api_key = vars.get("api_key", os.getenv("CUSTOM_LLM_API_KEY"))
    model = vars.get("model", os.getenv("CUSTOM_LLM_MODEL", "default"))
    post_proc_name = vars.get("post_processor", "default")  # e.g., "financial", "sentiment"
    
    # Select post-processor dynamically
    post_processors = {
        "default": default_post_processor,
        "financial": financial_post_processor,
        "sentiment": sentiment_post_processor,
        # Add more: e.g., "ner": ner_post_processor
    }
    post_processor = post_processors.get(post_proc_name, default_post_processor)
    
    processor = GenericLLMProcessor(
        base_url=base_url, 
        api_key=api_key, 
        model=model,
        post_processor=post_processor
    )
    
    # vars from PromptFoo
    result = processor.invoke(
        prompt_template=prompt,
        context=vars.get("context", ""),
        user_query=vars.get("query", ""),
        doc_path=vars.get("doc_path")
    )
    
    # Return as string (PromptFoo expects); JSON for easy assertion
    return json.dumps(result)

if __name__ == "__main__":
    # PromptFoo CLI compatibility
    input_data = json.loads(sys.stdin.read())
    output = run_prompt(input_data["prompt"], input_data["vars"])
    print(json.dumps({"output": output}))
