#!/usr/bin/env python3
"""
End-to-end SMS-spam model:
– download & extract dataset
– clean / preprocess
– vectorise (unigrams + bigrams)
– train MultinomialNB with GridSearchCV
– evaluate on hold-out set
– save pipeline
"""

# ─────────────────── imports ───────────────────
import os, io, re, zipfile, requests, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ─────────────────── constants ───────────────────
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
EXTRACT_DIR = "sms_spam_collection"
MODEL_PATH = "spam_detection_model.joblib"
SEED = 42

# ─────────────────── helpers ───────────────────
def download_and_extract(url: str, dest_dir: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(dest_dir)
    return os.path.join(dest_dir, "SMSSpamCollection")

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    df = df.drop_duplicates()
    return df

def preprocess_text(series: pd.Series) -> pd.Series:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    def _clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z\s$!]", "", text)
        tokens = word_tokenize(text)
        tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
        return " ".join(tokens)

    return series.apply(_clean)

def build_pipeline() -> Pipeline:
    vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
    clf = MultinomialNB()
    return Pipeline([("vectorizer", vectorizer), ("classifier", clf)])

# ─────────────────── main ───────────────────
if __name__ == "__main__":
    # 1. data
    data_file = download_and_extract(DATA_URL, EXTRACT_DIR)
    df = load_dataset(data_file)
    df["message"] = preprocess_text(df["message"])
    y = df["label"].map({"ham": 0, "spam": 1}).values
    X_text = df["message"]

    # 2. split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 3. pipeline & tuning
    pipe = build_pipeline()
    param_grid = {"classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # 4. evaluation
    y_pred = best_model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))

    # 5. save
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    # 6. quick demo
    demo = [
        "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
        "Hey, are we still meeting up for lunch today?",
    ]
    demo_pred = best_model.predict(demo)
    for msg, lbl in zip(demo, demo_pred):
        print(f"[{'SPAM' if lbl else 'HAM'}] {msg}")
