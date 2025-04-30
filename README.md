# 📧 SMS Spam Detector

A complete end-to-end SMS spam detection pipeline in a **single Python file** using the classic SMS Spam Collection dataset and a **Multinomial Naive Bayes** classifier.

---

## 🔍 Project Overview

This project builds a machine learning model that classifies SMS messages as **spam** or **ham (not spam)**. The process includes:

- ✅ Downloading and loading the SMS Spam Collection dataset  
- 🧹 Preprocessing messages (lowercasing, tokenization, stop word removal, stemming) using **NLTK**  
- 🧠 Converting text into feature vectors with **CountVectorizer** (unigrams and bigrams)  
- 🤖 Training a **Multinomial Naive Bayes** model with **GridSearchCV** for hyperparameter tuning  
- 📈 Evaluating model performance using precision, recall, F1-score, and confusion matrix  
- 💾 Saving the trained model as a `.joblib` file  
- 🧪 Predicting new SMS messages with confidence probabilities

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
