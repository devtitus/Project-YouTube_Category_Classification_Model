from transformers import DistilBertTokenizer, DistilBertModel

# Specify the model name
model_name = "distilbert-base-uncased"

# Download and load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Download and load the model
model = DistilBertModel.from_pretrained(model_name)

# Save the model and tokenizer to your local directory
model.save_pretrained("./distilbert-base-uncased")
tokenizer.save_pretrained("./distilbert-base-uncased")