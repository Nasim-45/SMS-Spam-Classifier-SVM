# SMS Spam Classifier (SVM)

A machine learning project that reads an SMS message and decides whether it is **Spam** (unwanted or fraudulent) or **Ham** (a normal message).

---

## 1. Problem
Spam SMS messages try to trick people with fake prizes, links and phone numbers. The goal is to detect them automatically with high accuracy, and above all to **not miss real spam** (high recall).

## 2. Dataset
- SMS Spam Collection (UCI / Kaggle): https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- Each row has a label (`spam` or `ham`) and the message text.
- The data is imbalanced: there are far more Ham messages than Spam.
- Download `spam.csv` and upload it in Colab as `SMS.csv`.

## 3. How the system works (pipeline)

1. **Load data**: read the CSV and keep two columns, `label` and `message`.
2. **Clean text**: lowercase the text, remove punctuation and stopwords (the, is, a ...), and lemmatize words (running -> run).
3. **TF-IDF**: turn text into numbers. A word gets a high score if it is frequent in a message but rare across all messages. Single words and two-word phrases (such as "call now") are both used.
4. **Extra features**: add hand-made features about the shape of the message (see section 4).
5. **Train the model**: a Linear SVM (LinearSVC) learns to separate Spam from Ham.
6. **Tune and evaluate**: pick the best settings, tune the spam threshold, run cross-validation and compare with other models.
7. **Predict**: classify new messages with `predict_sms()` and `predict_batch()`.

## 4. Extra features added (and why)

| Feature | What it measures | Why it helps |
|---|---|---|
| Character length | Length of the message | Spam is often longer |
| Word count | Number of words | Spam follows a different pattern |
| Digit count | Number of digits | Spam has prices, codes, phone numbers |
| Uppercase ratio | Share of capital letters | Spam often shouts: "WINNER!!!" |
| Currency count | Count of £ $ symbols | Money offers are common in spam |
| Exclamation count | Number of `!` | Spam is more excited |
| Has URL | Link present or not | Spam often contains links |
| Has spam word | free, win, prize, urgent, claim, offer ... | Strong spam keywords |
| Has long number | Phone-style long number | Spam asks you to call or text a number |

These features are scaled and combined with the TF-IDF features, so the model sees both **what words are used** and **how the message looks**.

## 5. Improvements over the basic version

### Class balancing
Ham is much more common than Spam, so a normal model tends to ignore Spam. With `class_weight='balanced'` a Spam mistake counts more, which raises spam recall.

### Probability output
The model gives a spam probability (for example 92%) instead of only a label. This uses `CalibratedClassifierCV`.

### Threshold tuning
Spam is usually decided at 50% probability. The best cut-off is chosen from the **Precision-Recall curve**, using out-of-fold predictions, so the balance between catching spam and avoiding false alarms is better.

### 5-fold stratified cross-validation
The data is split into 5 parts and the model is tested 5 times, each time on a different part. This gives a more reliable score than a single split. Accuracy, precision, recall and F1 are reported.

### Model comparison
Linear SVM is compared with Logistic Regression and Naive Bayes to show that SVM is a good choice.

### Evaluation graphs
- **Confusion matrix**: correct and wrong predictions.
- **ROC curve**: how well the model separates the two classes.
- **Precision-Recall curve**: the trade-off between precision and recall (important for imbalanced data).

### Error analysis
A table of messages the model got wrong, to see where it is weak.

### Top spam and ham words
The words with the largest model coefficients, showing which words push a message toward Spam and which toward Ham (explainability).

## 6. Using the model

**Single message**
```python
predict_sms("WINNER!! You won a free prize. Call 09061701461 now")
# returns: label, spam_probability, flags (which warning signs were found)
```

**Many messages**
```python
predict_batch(["Hi, are you coming for lunch?", "Claim your free cash now http://x.com"])
# returns a table and saves it to sms_predictions.csv
```

**Saved model**
The trained model and the tuned threshold are saved to `sms_spam_model.joblib`, so the model can be reused without training again.

## 7. Results
- Baseline model (before improvements): accuracy about 98.45%, spam precision 0.983, spam recall 0.893, spam F1 0.936.
- Improved model: run the notebook and write your own numbers here (accuracy, precision, recall, F1 and best threshold).

## 8. How to run
1. Open `SMS_Spam_SVM_Updated.ipynb` in Google Colab.
2. Run all cells from top to bottom.
3. Upload `SMS.csv` when asked.
4. Use `predict_sms()` to test your own messages.

## 9. Technologies
Python, scikit-learn, NLTK, pandas, NumPy, matplotlib, seaborn, joblib

## 10. Author
Nasim, Bangladesh University of Professionals
