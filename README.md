# 📱 Explainable SMS Spam Detector

An Explainable AI-powered SMS Spam Detection system that classifies text messages as **Spam** or **Ham (Legitimate)** using Machine Learning and Natural Language Processing (NLP). The project not only predicts whether a message is spam but also explains the factors influencing the prediction, improving transparency and trust in the model.

## 🚀 Project Highlights

* ✅ **98.3% Accuracy** on the held-out test set
* 🎯 **99.2% Precision** for spam detection
* 📩 Trained on **5,572 labelled SMS messages**
* 🔍 Explainable predictions with feature-based insights
* 🌐 Interactive web application built using Flask
* ⚡ TF-IDF vectorization with 3,000 features

---

## 📊 Dataset Overview

### SMS Spam Collection Dataset

| Metric           | Value         |
| ---------------- | ------------- |
| Total Messages   | 5,572         |
| Ham Messages     | 4,825 (86.6%) |
| Spam Messages    | 747 (13.4%)   |
| Test Samples     | 1,035         |
| Train-Test Split | 80/20         |
| Random State     | 2             |

### Data Cleaning

* Removed 403 duplicate records
* Cleaned and standardized text
* Final corpus contained 5,169 unique SMS messages

---

## 🧹 Text Preprocessing Pipeline

Every SMS message passes through the following NLP pipeline:

### 1️⃣ Lowercase Conversion

Converts text to lowercase to eliminate case sensitivity.

Example:

```
FREE PRIZE!!
↓
free prize!!
```

### 2️⃣ Tokenization (NLTK)

Splits text into individual words/tokens.

Example:

```
free prize!!
↓
["free", "prize"]
```

### 3️⃣ Alphanumeric Filtering

Removes punctuation and special characters while retaining meaningful words and numbers.

### 4️⃣ Stopword Removal

Removes common English words such as:

* the
* is
* and
* of
* to

### 5️⃣ Porter Stemming

Reduces words to their root form.

Examples:

```
winning → win
prizes → prize
```

### 6️⃣ TF-IDF Vectorization

Transforms text into numerical feature vectors using:

* Maximum Features: 3,000
* Sparse representation
* Importance-weighted word frequencies

---

## 🤖 Machine Learning Models Evaluated

| Model                   | Accuracy | Precision |
| ----------------------- | -------- | --------- |
| Bernoulli Naive Bayes   | 98.3%    | 99.2%     |
| Multinomial Naive Bayes | 97.1%    | 100%      |
| Gaussian Naive Bayes    | 87.4%    | 51.8%     |

### Final Model Selection

**Multinomial Naive Bayes** was selected because it achieved:

* 100% Precision
* Zero False Positives
* Strong overall performance

This ensures legitimate messages are never incorrectly flagged as spam.

---

## 📈 Model Evaluation

### Confusion Matrix

|             | Predicted Ham | Predicted Spam |
| ----------- | ------------- | -------------- |
| Actual Ham  | 896           | 0              |
| Actual Spam | 30            | 108            |

### Performance Metrics

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 97.1% |
| Precision | 100%  |
| Recall    | 78.3% |

### Interpretation

* No legitimate SMS was marked as spam.
* 30 spam messages were missed.
* Prioritized precision to minimize false alarms.

---

## 🚨 Top Spam Trigger Words

The model frequently identifies the following words as strong spam indicators:

```
free
call
win
prize
claim
txt
urgent
cash
mobile
text
awarded
stop
reply
tone
service
customer
selected
network
bonus
guaranteed
voucher
draw
```

---

## 🛠 Technology Stack

### Programming Language

* Python 3.13

### Machine Learning

* Scikit-learn

  * MultinomialNB
  * BernoulliNB
  * TfidfVectorizer

### Natural Language Processing

* NLTK

  * Tokenization
  * Stopword Removal
  * Porter Stemming

### Data Processing

* Pandas
* NumPy

### Backend

* Flask

### Model Storage

* Pickle

---

## 📂 Project Structure

```text
Explainable-SMS-Spam-Detector/
│
├── static/
│
├── templates/
│   └── index.html
│
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── app.py
├── train_model.py
├── spam.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Explainable-SMS-Spam-Detector.git
```

Navigate to the project directory:

```bash
cd Explainable-SMS-Spam-Detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

Open in browser:

```text
http://localhost:5000
```

---

## 🎯 Key Learning Outcomes

* Natural Language Processing (NLP)
* Text Classification
* Feature Engineering
* TF-IDF Vectorization
* Explainable AI (XAI)
* Machine Learning Model Evaluation
* Flask Web Development
* Model Serialization with Pickle

---

## 📸 Screenshots

Add screenshots of:

* Home Page
* Prediction Interface
* Model Performance Dashboard
* Dataset Analysis
* Spam Prediction Results

---

## 👩‍💻 Authors

Uma E Rubab

---

## 📄 License

This project is intended for educational and learning purposes.
