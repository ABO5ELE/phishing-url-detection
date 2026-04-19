# Phishing URL Detection Using Machine Learning

A machine learning project that detects phishing URLs using multiple classification algorithms.
Built as part of the Level 4 Primers course at European Universities in Egypt (EUE).

## Student
- Name: Yassin Khalid
- Course: Level 4 Primers
- University: European Universities in Egypt

## Project Overview
This project builds a system that automatically detects whether a URL is phishing or legitimate.
The system was trained on 11,430 URLs with 87 features each.
Random Forest achieved the best accuracy of 97.07%.

## Dataset
- Source: Kaggle - Web Page Phishing Detection Dataset
- Total URLs: 11,430
- Phishing: 5,715
- Legitimate: 5,715
- Missing values: 0

## Models Compared
| Model | Accuracy |
|---|---|
| Random Forest | 97.07% |
| Decision Tree | 93.92% |
| KNN | 83.60% |
| Logistic Regression | 82.72% |
| Naive Bayes | 76.29% |

## Folder Structure
phishing-url-detection/
├── data/
│   └── README.md
├── notebooks/
│   └── phishing_detection.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   └── test_model.py
├── config/
│   └── config.py
├── requirements.txt
└── README.md
