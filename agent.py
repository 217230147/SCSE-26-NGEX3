### COMPLETE THE CODE ###

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_tool import search_policy_tool

MODEL_PATH = "/mnt/c/Users/19550/Desktop/11/SCSE_26_GEX3/models/qwen3-0.6b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype="auto")

def ask_agent(question: str) -> str:
    context = search_policy_tool(question)
    
    system_prompt = (
        "You are a university IT support assistant. "
        "Answer the user's question ONLY using the supported university IT information provided below. "
        "If the requested fact (such as a phone number) is NOT available in the provided information, "
        "you MUST state clearly that you do not know and that the information is not specified. "
        "DO NOT invent or hallucinate any phone number or external instructions.\n\n"
        f"University Information Context:\n{context}\n\n"
    )
    
    prompt = (
        f"{system_prompt}"
        f"User: {question}\n\n"
        "Assistant:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )
        
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    answer = response.split("Assistant:")[-1].strip()
    return answer

if __name__ == "__main__":
    test_q = "What exact phone number should I call for the university IT Service Desk?"
    print(f"Question: {test_q}")
    print(f"Answer: {ask_agent(test_q)}")