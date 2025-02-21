from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # Allow frontend to interact with backend

# Load NLP model for fake news detection
nlp = spacy.load("en_core_web_sm")
fact_checker = pipeline("text-classification", model="facebook/bart-large-mnli")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")

    # Simple NLP Preprocessing
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Fake News Classification
    result = fact_checker(text)
    
    response = {
        "original_text": text,
        "entities": entities,
        "fact_check_result": result
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
