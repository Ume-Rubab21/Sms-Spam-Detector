from flask import Flask, render_template, request, jsonify
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('models/model.pkl', 'rb'))
tfidf = pickle.load(open('models/vectorize.pkl', 'rb'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered = [t for t in tokens if t.isalnum()]
    filtered = [t for t in filtered if t not in stopwords.words('english') and t not in string.punctuation]
    stemmed = [ps.stem(t) for t in filtered]
    return " ".join(stemmed)

def get_spam_words(text):
    """Find which words in the SMS are strong spam indicators."""
    tokens = nltk.word_tokenize(text.lower())
    words = [t for t in tokens if t.isalnum()]
    
    vocab = tfidf.vocabulary_
    feature_names = tfidf.get_feature_names_out()
    
    # Get feature log probabilities from the model (MultinomialNB)
    spam_class_idx = list(model.classes_).index(1)
    ham_class_idx = list(model.classes_).index(0)
    
    spam_words = []
    for word in words:
        stemmed = ps.stem(word)
        if stemmed in vocab:
            feat_idx = vocab[stemmed]
            spam_prob = model.feature_log_prob_[spam_class_idx][feat_idx]
            ham_prob = model.feature_log_prob_[ham_class_idx][feat_idx]
            score = spam_prob - ham_prob  # higher = more spam-like
            if score > -1.0:  # threshold: word leans spam
                spam_words.append({'word': word, 'score': round(float(score), 3)})
    
    # Sort by score descending
    spam_words.sort(key=lambda x: x['score'], reverse=True)
    return spam_words[:10]  # top 10 spam-indicator words

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sms = data.get('sms', '').strip()
    
    if not sms:
        return jsonify({'error': 'No message provided'}), 400
    
    # Preprocess
    transformed = transform_text(sms)
    
    # Vectorize
    vector = tfidf.transform([transformed])
    
    # Predict
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    
    spam_idx = list(model.classes_).index(1)
    ham_idx = list(model.classes_).index(0)
    
    spam_prob = round(float(probabilities[spam_idx]) * 100, 1)
    ham_prob = round(float(probabilities[ham_idx]) * 100, 1)
    
    label = 'spam' if prediction == 1 else 'ham'
    
    # Get spam indicator words
    spam_words = get_spam_words(sms)
    
    # Generate reason
    if label == 'spam':
        if spam_words:
            top = [w['word'] for w in spam_words[:3]]
            reason = f"This message contains spam trigger words like <b>{', '.join(top)}</b>. Spam messages often use urgent language, prize offers, or call-to-action phrases to lure recipients."
        else:
            reason = "The overall pattern and structure of this message closely matches known spam patterns in our training data."
    else:
        reason = "This message reads like a normal, personal conversation. It lacks the typical markers of spam such as urgency, prizes, links, or promotional language."
    
    return jsonify({
        'label': label,
        'spam_prob': spam_prob,
        'ham_prob': ham_prob,
        'spam_words': spam_words,
        'reason': reason,
        'transformed': transformed
    })

if __name__ == '__main__':
    app.run(debug=True)