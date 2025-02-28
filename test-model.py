from transformers import AutoTokenizer, AutoModel
import torch

# Load tokenizer and model from local directory
model_path = "./tinybert_model"  # Change this if needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

print("Model loaded successfully!")


text = "This channel features the latest football match highlights and sports analysis."

# Tokenize input text
inputs = tokenizer(text, return_tensors="pt")

# Run the model (get embeddings)
with torch.no_grad():
    outputs = model(**inputs)

print("Model ran successfully! Output shape:", outputs.last_hidden_state.shape)