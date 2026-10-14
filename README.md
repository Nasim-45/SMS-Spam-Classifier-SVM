# SMS Spam Classifier (SVM)

Detects spam SMS messages using TF-IDF features and a Linear SVM.

## Features
- Text cleaning: stopword removal and lemmatization
- TF-IDF with unigrams and bigrams
- Extra features: message length, word count, digits, uppercase ratio, currency symbols, exclamation marks, URLs, spam keywords, long numbers
- Class balancing for better spam recall
- Spam probability output
- Threshold tuning with precision-recall curves
- 5-fold stratified cross-validation
- Model comparison: SVM vs Logistic Regression vs Naive Bayes
- ROC and PR curves, confusion matrix, error analysis
- Top spam and ham indicator words
- `predict_sms()` and `predict_batch()` functions
- Saved model with joblib

## Dataset
[SMS Spam Collection (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
Download `spam.csv` and upload it in Colab as `SMS.csv`.

## How to run
1. Open `SMS_Spam_SVM_Updated.ipynb` in Google Colab.
2. Run all cells in order.
3. Upload the dataset when asked.

## Tech
Python, scikit-learn, NLTK, pandas, matplotlib, seaborn

## Author
Md Nasimuzzaman Emon
