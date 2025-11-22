import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import json
import re
import random
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# -----------------------------
# 1. Load model and tokenizer
# -----------------------------
tok_gpt2 = AutoTokenizer.from_pretrained("distilgpt2")
model_gpt2 = AutoModelForCausalLM.from_pretrained("distilgpt2")


# -----------------------------
# 2.1 Base prompt
# -----------------------------
BASE_PROMPT = (
    'You are given a purchase request. Extract a JSON object with fields item and quantity.\n'
    'Text: "Order three boxes of blue markers for the design team."\n'
    'JSON:'
)

# -----------------------------
# 2.2 Schema prompt
# -----------------------------
SCHEMA_PROMPT = (
    'You are given a purchase request. Extract a JSON object with fields item and quantity.\n'
    'Text: "Order three boxes of blue markers for the design team."\n'
    'Output must be valid JSON exactly: {"item": "<string>", "quantity": <integer>}. No commentary.\n'
    'JSON:'
)

# -----------------------------
# Helper: generate one sample
# -----------------------------
def generate_one_sample(model, tokenizer, prompt, decode_kwargs):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=60,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True if decode_kwargs.get("temperature", 0) > 0 else False,
        **decode_kwargs
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()


# -----------------------------
# Helper: generate 10 samples
# -----------------------------
def generate_10_samples(model, tokenizer, prompt, decode_kwargs, config_name=""):
    samples = []
    for i in range(10):
        sample = generate_one_sample(model, tokenizer, prompt, decode_kwargs)
        samples.append(sample)
        if config_name:
            print(f"{config_name}: {i+1}/10", end='\r', flush=True)
    if config_name:
        print()  
    return samples

# -----------------------------
# Metrics computation functions
# -----------------------------
def compute_distinct_n(samples, n):
    """Compute distinct-n: number of unique n-grams / total n-grams"""
    all_ngrams = []
    for sample in samples:
        # Tokenize into words (split on whitespace and punctuation)
        tokens = re.findall(r'\b\w+\b', sample.lower())
        # Generate n-grams
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            all_ngrams.append(ngram)
    
    if len(all_ngrams) == 0:
        return 0.0
    
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0

def compute_mean_length(samples):
    """Compute mean output length in tokens"""
    lengths = []
    for sample in samples:
        # Count tokens (words)
        tokens = re.findall(r'\b\w+\b', sample)
        lengths.append(len(tokens))
    return sum(lengths) / len(lengths) if lengths else 0.0

def compute_repetition_rate(samples):
    """Compute repetition rate: fraction of adjacent identical tokens"""
    total_adjacent_pairs = 0
    identical_pairs = 0
    
    for sample in samples:
        tokens = re.findall(r'\b\w+\b', sample.lower())
        for i in range(len(tokens) - 1):
            total_adjacent_pairs += 1
            if tokens[i] == tokens[i+1]:
                identical_pairs += 1
    
    return identical_pairs / total_adjacent_pairs if total_adjacent_pairs > 0 else 0.0

def compute_json_validity_rate(samples):
    """Compute JSON validity rate: fraction that parses correctly and has both item and quantity fields"""
    valid_count = 0
    
    for sample in samples:
        try:
            # Try to find JSON object in the sample
            # Look for opening brace and try to parse from there
            start_idx = sample.find('{')
            if start_idx != -1:
                # Try to find the matching closing brace by parsing
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(sample)):
                    if sample[i] == '{':
                        brace_count += 1
                    elif sample[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if brace_count == 0:
                    json_str = sample[start_idx:end_idx]
                    parsed = json.loads(json_str)
                    
                    # Check if it has both 'item' and 'quantity' fields
                    if isinstance(parsed, dict) and 'item' in parsed and 'quantity' in parsed:
                        valid_count += 1
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            pass
    
    return valid_count / len(samples) if samples else 0.0

def compute_all_metrics(samples):
    """Compute all metrics for a set of samples"""
    return {
        'distinct-1': compute_distinct_n(samples, n=1),
        'distinct-2': compute_distinct_n(samples, n=2),
        'mean_length': compute_mean_length(samples),
        'repetition_rate': compute_repetition_rate(samples),
        'json_validity_rate': compute_json_validity_rate(samples)
    }

# -----------------------------
# 3. Decoding configurations
# -----------------------------
decoding_setups = {
    "greedy": {"temperature": 0.0},
    "temp_0.7": {"temperature": 0.7},
    "temp_1.0": {"temperature": 1.0},
    "topk_40": {"temperature": 0.7, "top_k": 40},
    "topk_200": {"temperature": 0.7, "top_k": 200},
    "topp_0.8": {"temperature": 0.7, "top_p": 0.8},
    "topp_0.95": {"temperature": 0.7, "top_p": 0.95},
}

# -----------------------------
# Run each configuration
# -----------------------------
all_outputs = {}
all_metrics = {}

for name, params in decoding_setups.items():
    print(f"\n=== Generating for: {name} ===")
    base_samples = generate_10_samples(model_gpt2, tok_gpt2, BASE_PROMPT, params, config_name=name)
    schema_samples = generate_10_samples(model_gpt2, tok_gpt2, SCHEMA_PROMPT, params, config_name=name)
    all_outputs[name] = {
        "base": base_samples,
        "schema": schema_samples
    }
    # Compute metrics
    metrics_base = compute_all_metrics(base_samples)
    metrics_schema = compute_all_metrics(schema_samples)
    metrics = {
        "base": metrics_base,
        "schema": metrics_schema
    }
    all_metrics[name] = {
        "base": metrics_base,
        "schema": metrics_schema
    }
    

# Print summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Configuration':<15} {'Distinct-1':<12} {'Distinct-2':<12} {'Mean Len':<10} {'Rep Rate':<10} {'JSON Valid':<10}")
print("-"*80)
for name in decoding_setups.keys():
    m = all_metrics[name]
    print(f"{'base_'+name:<15} {m['base']['distinct-1']:<12.4f} {m['base']['distinct-2']:<12.4f} {m['base']['mean_length']:<10.2f} {m['base']['repetition_rate']:<10.4f} {m['base']['json_validity_rate']:<10.4f}")
    print(f"{'schema_'+name:<15} {m['schema']['distinct-1']:<12.4f} {m['schema']['distinct-2']:<12.4f} {m['schema']['mean_length']:<10.2f} {m['schema']['repetition_rate']:<10.4f} {m['schema']['json_validity_rate']:<10.4f}")
# Save all samples to a file
output_file = "all_samples.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_outputs, f, indent=2, ensure_ascii=False)
print(f"\nAll samples saved to: {output_file}")




# -----------------------------
# Conclusion & Comments
# -----------------------------
# Across decoding strategies, increasing temperature, top-k, and top-p consistently raised diversity (higher distinct-1/2), but also introduced semantic drift from the intended JSON structure. 
# Greedy decoding produced the most repetitive and least diverse outputs, yet still failed to generate valid JSON due to the base prompt being too weak. 
# Adding the schema-constrained prompt significantly reduced drift and shortened generations, but validity remained near zero, indicating that distilGPT-2 struggles with structured extraction without additional fine-tuning. 
# Top-k and top-p with schema prompts achieved the best balance between diversity and stability, though still not sufficient for correctness. 
# A natural next step would be to apply constrained decoding (e.g., JSON grammar, regex-guided generation) or post-validation with iterative repair to reliably enforce schema compliance.
