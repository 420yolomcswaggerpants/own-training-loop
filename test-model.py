from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "./nimbus-own-loop"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

questions = [
    "What coffees do you sell?",
    "How much is shipping?",
    "Do you sell cold brew?",
    "What's your most popular drink?"
]

for question in questions:
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.3,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:\n")[-1].strip()
    print(f"Q: {question}")
    print(f"A: {response}")
    print("---")