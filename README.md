# 📄 NLP - Sentiment Analysis Project 
[Open SentiScope — AI Sentiment Analyzer](https://sentiscoreanalyzer.streamlit.app/)
## Project Overview

**Project Name:** SentiScope — AI Sentiment Analyzer

**Description:**
This project is a Machine Learning-based **Sentiment Analysis Tool**. It analyzes text data and predicts the sentiment as **Positive**, **Neutral**, or **Negative**.

The goal is to classify user reviews or text data into sentiments with high accuracy using multiple machine learning models and select the **best performing model** for deployment.

---

## 🚀 Features

1. Classifies text into **Positive, Neutral, or Negative**.
2. Uses **multiple ML models**:

   * Logistic Regression
   * Support Vector Machine (SVM) — FINAL MODEL
   * Naive Bayes
   * XGBoost
3. Provides **accuracy, precision, recall, and F1-score** for evaluation.
4. **TF-IDF vectorizer** is used for converting text to numeric features.
5. All models and vectorizer are **saved in a single folder (`Models`)** for reuse.
6. **Deployment ready**: Streamlit app (`app.py`) allows users to input text and see sentiment prediction.

---

## 🛠️ Technology Stack

* **Language:** Python 3.12
* **Libraries:**

  * `pandas` — Data handling
  * `numpy` — Numerical computation
  * `scikit-learn` — Machine learning models and metrics
  * `xgboost` — XGBoost classifier
  * `joblib` / `pickle` — Save/load models
  * `streamlit` — Deployment and interactive UI

---

## 📂 Project Structure

```
NLP_Sentiment_Analysis/
│
├─ app.py                # Streamlit deployment file
├─ data/
│   └─ Raw_Dataset.csv   # Original raw dataset
├─ models/
│   ├─ FINAL_MODEL_SVM.pkl
│   ├─ FINAL_MODEL_LR.pkl
│   ├─ FINAL_MODEL_NB.pkl
│   ├─ FINAL_MODEL_XGB.pkl
│   └─ tfidf_vectorizer.pkl
├─ preprocessing/
│   └─ preprocessing_script.ipynb
├─ Stage6_Model_Building/
│   └─ stage6_model_building.ipynb
├─ requirements.txt
└─ README.md
```

---

## 📝 Step-by-Step Workflow

### **Stage 1: Data Loading**

1. Load the raw dataset (`Raw_Dataset.csv`) using **pandas**.
2. Check columns (`title`, `body`, `rating`) and missing values.
3. Inspect **distribution of ratings** for class imbalance.

---

### **Stage 2: Preprocessing**

1. Combine relevant text columns (e.g., `title + body`) for feature.
2. Convert `rating` to **sentiment labels**:

   * Positive (rating ≥ 4)
   * Neutral (rating = 3)
   * Negative (rating ≤ 2)
3. Clean text:

   * Lowercase
   * Remove punctuation, numbers, special characters
   * Remove stopwords
   * Optional: Lemmatization/stemming
4. Check class distribution. Apply **class balancing** using oversampling if required.

---

### **Stage 3: Feature Engineering**

1. Use **TF-IDF vectorizer** to convert text into numeric features.
2. Fit TF-IDF on training data, transform both train and test sets.

---

### **Stage 4: Train/Test Split**

1. Split data into **training and testing sets** (e.g., 80:20).
2. Ensure **stratified split** to maintain class distribution.

---

### **Stage 5: Model Building (Stage 6 in project)**

1. **Logistic Regression**

   * Hyperparameter tuning: `C`
   * Evaluated using **accuracy, precision, recall, F1-score**

2. **Support Vector Machine (SVM)** — **Final Model**

   * Hyperparameter: `C` and `class_weight='balanced'`
   * Best performing model for deployment

3. **Naive Bayes**

   * `MultinomialNB` for text classification

4. **XGBoost**

   * `XGBClassifier` with label encoding
   * Trained for multi-class sentiment classification

5. **Evaluation**

   * Metrics: Accuracy, Precision, Recall, F1-score
   * Compare models → **SVM chosen as final model**

---

### **Stage 6: Save Models**

* All models and TF-IDF vectorizer saved using `pickle` inside **`Models/` folder**:

```python
# Example: Save SVM as final model
with open('Models/FINAL_MODEL_SVM.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Save TF-IDF vectorizer
with open('Models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
```

---

### **Stage 7: Deployment**

1. **Streamlit App (`app.py`)**:

   * Input box for user text
   * Loads **final SVM model** and **TF-IDF vectorizer**
   * Predicts and displays sentiment with confidence scores

2. **Run App**:

```bash
streamlit run app.py
```

---

## ✅ Final Model Results

| Model               | Accuracy | Macro F1 | Weighted F1 | Notes                   |
| ------------------- | -------- | -------- | ----------- | ----------------------- |
| Logistic Regression | 0.7986   | 0.68     | 0.78        | Good baseline           |
| **SVM (Final)**     | 0.8056   | 0.67     | 0.78        | Best overall model ✅    |
| Naive Bayes         | 0.7604   | 0.54     | 0.70        | Lower recall on Neutral |
| XGBoost             | 0.7639   | 0.61     | 0.74        | Slightly lower than SVM |

**Best Model:** **SVM**

* **Hyperparameters:** `C=1, class_weight='balanced'`
* High precision and recall for Positive & Negative classes.
* Balanced performance on imbalanced data.

---

## 📌 Notes

* Neutral class has fewer samples → harder to predict.
* SVM with `class_weight='balanced'` handles imbalance better.
* All models can be retrained by running **Stage6 notebook**.
* TF-IDF vectorizer must be loaded with final model for deployment.

---

## 📦 Requirements

`requirements.txt` should include:

```
pandas
numpy
scikit-learn
xgboost
streamlit
joblib
```

---

## 💻 How to Run

1. Clone repository:

```bash
git clone <repo_url>
cd Sentiment_Analysis_Project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Streamlit app:

```bash
streamlit run app.py
```

4. Enter text → get predicted sentiment.

---

✅ **Project Complete!**
All preprocessing, model building, evaluation, and deployment steps are included.


