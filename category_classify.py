from fastapi import FastAPI, HTTPException, Query
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI()

MODEL_PATH = "./fine_tuned_tinybert"  # Path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # Ensure model is in evaluation mode

@app.post("/predict")
def predict_category(text: str = Query(..., description="The YouTube channel description")):
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1)[0, predicted_class].item()
        
        # Use the model's id2label mapping
        predicted_category = model.config.id2label.get(str(predicted_class), model.config.id2label.get(predicted_class))
        
        return {"category": predicted_category, "confidence": round(confidence, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
