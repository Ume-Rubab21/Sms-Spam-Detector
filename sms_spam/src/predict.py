import pickle
import os

# Use absolute paths or relative to the script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorize.pkl")
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

# Load vectorizer and model
vectorizer = pickle.load(open(vectorizer_path, "rb"))
model = pickle.load(open(model_path, "rb"))

def predict_spam(text):
    vector = vectorizer.transform([text])
    result = model.predict(vector)
    return int(result[0]) # Returns 0 or 1
