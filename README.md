# Newsgroup Baseline Comparison

A comprehensive comparison of four baseline classification methods on the complete 20 newsgroups text classification task.

## Overview

This project implements and evaluates four classical machine learning baselines for multi-class text classification:

1. **Count + Multinomial Naive Bayes** - CountVectorizer + probabilistic classifier
2. **Count + Logistic Regression** - CountVectorizer + linear classifier
3. **TFIDF + Multinomial Naive Bayes** - TfidfVectorizer + probabilistic classifier
4. **TFIDF + Linear SVM** - TfidfVectorizer + margin-based classifier

## Dataset

- **Source**: 20 Newsgroups (sklearn.datasets)
- **Categories**: All 20 categories
- **Total Documents**: 18,846 documents
- **Train/Test Split**: 80/20 (stratified)

## Feature Engineering

- **Vectorization**: Count and TF-IDF variants
- **Preprocessing**: Default sklearn preprocessing

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running the Comparison

```bash
python run.py
```

This will:
1. Load the 20 newsgroups dataset (all 20 categories, 18,846 documents)
2. Preprocess and vectorize text data
3. Train all four baseline models
4. Evaluate and compare performance
5. Display detailed results and summary statistics

## One-line Reproducible Run

From the project directory, run:

```bash
pip install -r requirements.txt && python run.py
```

## Results

The script outputs:
- **Accuracy**: Percentage of correct predictions
- **Macro F1**: Harmonic mean of precision and recall (macro-averaged)
- **Training Time**: Time taken to train each model

## Expected Performance

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Count + MultinomialNB | ~0.62 | ~0.59 |
| Count + LogisticRegression | ~0.68 | ~0.67 |
| TFIDF + MultinomialNB | ~0.68 | ~0.66 |
| TFIDF + LinearSVC | ~0.77 | ~0.76 |

## Key Insights

1. **TF-IDF outperforms raw counts** across most models
2. **LinearSVC with TF-IDF achieves best performance** (~77% accuracy)
3. **Stratified split ensures reproducibility** across class distributions
4. **All 20 categories** provide comprehensive benchmark

## Project Structure

```
newsgroup_baseline_comparison/
├── run.py          # Main script with analysis and implementation
├── requirements.txt # Python dependencies
└── README.md        # This file
├── run_output.log 
```

## Dependencies

- scikit-learn >= 1.0.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0

## Author

CS 265 Homework Assignment - Pacific University Spring 2025

## License

Educational Use Only

