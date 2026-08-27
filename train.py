import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# Tokenize
def tokenize_function(examples):
    texts = []
    for i in range(len(examples["prompt"])):
        text = f"### Instruction:\n{examples['prompt'][i]}\n\n### Response:\n{examples['completion'][i]}"
        texts.append(text)
    return tokenizer(texts, truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Set up optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
model.train()
num_epochs = 3
accumulation_steps = 8

optimizer.zero_grad()

for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(tokenized_dataset), 1):  # batch size 1
        batch = tokenized_dataset[i:i+1]
        
        input_ids = torch.tensor(batch["input_ids"])
        attention_mask = torch.tensor(batch["attention_mask"])
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        
        loss = outputs.loss / accumulation_steps
        
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}: average loss = {avg_loss:.4f}")

optimizer.step()
optimizer.zero_grad()

# Save model
model.save_pretrained("./nimbus-own-loop")
tokenizer.save_pretrained("./nimbus-own-loop")