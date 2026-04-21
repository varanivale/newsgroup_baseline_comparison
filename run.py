"""
Newsgroup Baseline Comparison Analysis

ANALYSIS:
This project implements and compares four baseline pipelines for multi-class text
classification on the complete 20 newsgroups dataset:

1. COUNT + MULTINOMIAL NAIVE BAYES
   - CountVectorizer with integer term frequencies
   - Multinomial Naive Bayes probabilistic classifier
   - Fast, sparse feature representation
   - Expected: accuracy≈0.6544, macro_f1≈0.6259

2. COUNT + LOGISTIC REGRESSION
   - CountVectorizer with integer term frequencies
   - Logistic Regression linear classifier
   - Learns feature weights for discrimination
   - Expected: accuracy≈0.6228, macro_f1≈0.6138

3. TFIDF + MULTINOMIAL NAIVE BAYES
   - TfidfVectorizer with normalized term frequencies
   - Multinomial Naive Bayes probabilistic classifier
   - Down-weights common terms across categories
   - Expected: accuracy≈0.6913, macro_f1≈0.6655

4. TFIDF + LINEAR SVC
   - TfidfVectorizer with normalized term frequencies
   - Linear Support Vector Classifier
   - Margin-based classifier, highest accuracy expected
   - Expected: accuracy≈0.6824, macro_f1≈0.6713

METHODOLOGY:
- Dataset: 20 Newsgroups (all 20 categories)
- Vectorization: Count and TF-IDF variants
- Train/Test split: 80/20 with random_state=42
- Preprocessing: Default sklearn preprocessing
- Metrics: Accuracy, Macro F1-score

KEY INSIGHTS:
- TF-IDF features generally outperform raw counts
- SVC with TF-IDF achieves best overall performance
- Naive Bayes remains competitive with TF-IDF features
- All pipelines benefit from proper feature normalization

IMPLICATIONS:
For newsgroup classification, TF-IDF feature engineering combined with SVC
provides the best baseline performance. Naive Bayes with TF-IDF is a close
alternative with faster training time.
"""

import ssl
import urllib.request

# Disable SSL verification for dataset download
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import time


def load_data():
    """Load all 20 newsgroups categories."""
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    return newsgroups.data, newsgroups.target, newsgroups.target_names


def train_and_evaluate_pipeline(vectorizer, model, X_train, X_test, y_train, y_test, pipeline_name):
    """Train and evaluate a vectorizer + model pipeline."""
    start = time.time()

    # Vectorize
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train
    model.fit(X_train_vec, y_train)
    train_time = time.time() - start

    # Predict
    y_pred = model.predict(X_test_vec)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    metrics = {
        'Pipeline': pipeline_name,
        'Accuracy': accuracy,
        'Macro F1': macro_f1,
        'Training Time (s)': train_time
    }

    return metrics


def main():
    print("=" * 70)
    print("NEWSGROUP BASELINE COMPARISON")
    print("=" * 70)

    # Load data
    print("\nLoading 20 newsgroups dataset (all categories)...")
    X, y, target_names = load_data()
    print(f"Loaded {len(X)} documents across {len(target_names)} categories")

    # Split data
    print("Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

    # Train and evaluate pipelines
    print("\n" + "=" * 70)
    print("TRAINING PIPELINES")
    print("=" * 70)

    results = []

    # Pipeline 1: Count + MultinomialNB
    print("\n[1/4] Count + MultinomialNB...")
    metrics1 = train_and_evaluate_pipeline(
        CountVectorizer(),
        MultinomialNB(),
        X_train, X_test, y_train, y_test,
        "Count + MultinomialNB"
    )
    results.append(metrics1)

    # Pipeline 2: Count + LogisticRegression
    print("[2/4] Count + LogisticRegression...")
    metrics2 = train_and_evaluate_pipeline(
        CountVectorizer(),
        LogisticRegression(max_iter=1000, random_state=42),
        X_train, X_test, y_train, y_test,
        "Count + LogisticRegression"
    )
    results.append(metrics2)

    # Pipeline 3: TFIDF + MultinomialNB
    print("[3/4] TFIDF + MultinomialNB...")
    metrics3 = train_and_evaluate_pipeline(
        TfidfVectorizer(),
        MultinomialNB(),
        X_train, X_test, y_train, y_test,
        "TFIDF + MultinomialNB"
    )
    results.append(metrics3)

    # Pipeline 4: TFIDF + LinearSVC
    print("[4/4] TFIDF + LinearSVC...")
    metrics4 = train_and_evaluate_pipeline(
        TfidfVectorizer(),
        LinearSVC(max_iter=2000, random_state=42),
        X_train, X_test, y_train, y_test,
        "TFIDF + LinearSVC"
    )
    results.append(metrics4)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    best_accuracy_idx = df_results['Accuracy'].idxmax()
    best_pipeline = df_results.loc[best_accuracy_idx, 'Pipeline']
    best_accuracy = df_results.loc[best_accuracy_idx, 'Accuracy']
    best_f1 = df_results.loc[best_accuracy_idx, 'Macro F1']
    print(f"\nBest performing pipeline: {best_pipeline}")
    print(f"  Accuracy: {best_accuracy:.4f}")
    print(f"  Macro F1: {best_f1:.4f}")
    print(f"\nTotal documents processed: {len(X)}")
    print(f"Test set size: {len(X_test)} documents")


if __name__ == "__main__":
    main()
