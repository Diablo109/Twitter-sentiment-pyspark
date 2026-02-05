# Twitter Sentiment Analysis using PySpark

## Overview
This project performs sentiment analysis on Twitter data using **PySpark**.  
It processes large-scale text data, applies preprocessing techniques, and trains multiple machine learning models to classify tweets as positive or negative.

The goal of this project is to demonstrate:
- Distributed data processing with PySpark
- Text preprocessing techniques
- Feature engineering using TF-IDF
- Model comparison and evaluation

---

## Features
- Large-scale tweet processing using PySpark
- Text cleaning and preprocessing
- Tokenization, stopword removal, and stemming
- N-gram feature generation
- TF-IDF vectorization
- Multiple model training and comparison
- Evaluation using F1-score and ROC-AUC
- Real-time sentiment prediction using Spark Structured Streaming

---

## Tech Stack
- Python
- PySpark
- Scikit-learn
- Pandas
- NumPy

---

## Models Used
The following classification models were trained and compared:

- Logistic Regression
- Naive Bayes
- Linear Support Vector Classifier (Linear SVC)

---

## Model Performance
| Model                | F1 Score | ROC-AUC |
|----------------------|----------|---------|
| Logistic Regression  | 0.67     | 0.76    |
| Naive Bayes          | 0.64     | 0.55    |
| Linear SVC           | 0.66     | 0.75    |

**Best Model:** Logistic Regression

---

## Real-time Sentiment Streaming
The trained model is used in a **Spark Structured Streaming** pipeline to perform real-time sentiment prediction on incoming tweets.

### Real-time Pipeline Steps
1. Spark reads streaming text data.
2. Incoming tweets are cleaned and preprocessed.
3. The trained best model predicts sentiment.
4. Predictions are displayed in real-time.

This demonstrates the ability to:
- Combine machine learning with streaming data
- Perform real-time inference using Spark

---

## Project Structure
```
twitter-sentiment-pyspark/
│
├── twitter_sentiment.ipynb   # Main project notebook
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/twitter-sentiment-pyspark.git
cd twitter-sentiment-pyspark
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage
1. Open the notebook:
```bash
jupyter notebook twitter_sentiment.ipynb
```
2. Run all cells to:
   - Load data
   - Preprocess text
   - Train models
   - Evaluate results

---

## Future Improvements
- Hyperparameter tuning using cross-validation
- Deep learning models (LSTM, BERT)
- Deployment as a web API

---

## Author
**Rahul**  
Aspiring Machine Learning Engineer  
