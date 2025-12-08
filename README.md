# 📰 Fake-News-Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)  
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)  
[![GitHub issues](https://img.shields.io/github/issues/Sahil-Shrivas/Fake-News-Prediction)](https://github.com/Sahil-Shrivas/Fake-News-Prediction/issues)

A machine learning project to **detect fake news** from text data. This model classifies news articles as **real** or **fake** using natural language processing (NLP) techniques and supervised ML algorithms.

---

## 📖 Overview

Fake news spreads misinformation and can influence public opinion. This project aims to **automate the detection of fake news** by analyzing text content using ML.  

The workflow includes:  

- Data preprocessing and cleaning  
- Text vectorization with TF-IDF  
- Training supervised ML models (Logistic Regression, Random Forest, etc.)  
- Evaluating performance with accuracy, precision, recall, F1-score  
- Optional deployment via a web interface for live predictions  

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python  
- **Libraries & Tools:**  
  - `pandas`, `numpy` — data manipulation  
  - `scikit-learn` — model training & evaluation  
  - `nltk`, `re` — text preprocessing  
  - `pickle` — save/load trained models  
  - `streamlit` (optional) — interactive web app  

> See `requirements.txt` for full dependencies.

---

## 📂 Dataset

- The project uses a **news dataset** containing labeled articles (Fake / Real).  
- Features include: `title`, `text`, and `label`.  
- Dataset can be found in the `data/` folder or loaded via external source if applicable.  

> ⚠️ Ensure data privacy when using any real news data.

---

## 📂 Project Structure

    Fake-News-Prediction/
    │── data/ # Dataset (CSV or processed files)
    │── Model_Training.ipynb # Notebook for data exploration & model training
    │── app.py # Streamlit web app for live predictions
    │── model.pkl # Trained ML model
    │── vector.pkl # TF-IDF vectorizer
    │── requirements.txt # Python dependencies
    │── README.md # This documentation
    │── LICENSE # MIT License

---

## 📊 Model Functionality

- Cleans and preprocesses text (removes punctuation, stopwords, tokenization)

- Converts text to numerical features using TF-IDF vectorization

- Trains ML models to classify news as Fake or Real

- Evaluates models using Accuracy, Precision, Recall, and F1-score

⚠️ Note: This model is for educational purposes. Predictions are not guaranteed to be accurate and should not replace professional fact-checking.

---

## ✅ Future Improvements

- Implement deep learning models (LSTM, BERT) for higher accuracy

- Use additional datasets from multiple sources

- Enhance the Streamlit web app with interactive visualizations

- Provide confidence scores and explainable predictions

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
    git clone https://github.com/Sahil-Shrivas/Fake-News-Prediction.git
    cd Fake-News-Prediction

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt

3. **Run the web app**
   ```bash
   streamlit run app.py
   
---

## 📬 Contact

- Author: Sahil Shrivas
- GitHub: https://github.com/Sahil-Shrivas
    

