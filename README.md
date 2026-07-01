# Fake vs Real News Headline Classification

## Overview

The **Fake vs Real News Headline Classification** project is a Machine Learning application that automatically classifies news headlines as **Fake** or **Real**. It uses Natural Language Processing (NLP) techniques to clean and process text data before training machine learning models. The project helps demonstrate how AI can be used to identify misleading news and reduce the spread of misinformation.

---

## Objectives

- Classify news headlines into **Fake** or **Real** categories.
- Apply NLP techniques for text preprocessing.
- Convert text into numerical features using **TF-IDF Vectorization**.
- Train and evaluate multiple Machine Learning models.
- Compare model performance using various evaluation metrics.

---

## Features

- Automatic fake news headline detection.
- Text preprocessing and cleaning.
- TF-IDF feature extraction.
- Model training using:
  - Multinomial Naive Bayes
  - Decision Tree Classifier
- 5-Fold Cross Validation.
- Performance evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- Saves trained models and evaluation reports.

---

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## Project Structure

```text
ML-Project2/
│
├── Datasets/
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── train_data_cleaned.csv
│   ├── big_test_data.txt
│   └── small_test_data.txt
│
├── results/
│   ├── train_results/
│   └── test_results/
│
├── model_train.py
├── model_test.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Fake-vs-Real-News-Headline-Classification.git
```

### 2. Navigate to the project folder

```bash
cd Fake-vs-Real-News-Headline-Classification
```

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train the Model

```bash
python model_train.py
```

This script:

- Loads the training dataset.
- Cleans invalid data.
- Preprocesses news headlines.
- Converts text into TF-IDF features.
- Trains Naive Bayes and Decision Tree models.
- Performs 5-Fold Cross Validation.
- Saves evaluation reports and trained models.

---

### Test the Model

```bash
python model_test.py
```

This script:

- Loads the testing dataset.
- Uses the trained models.
- Predicts whether each headline is Fake or Real.
- Generates accuracy scores, confusion matrices, and classification reports.

---

## Dataset

The datasets are stored inside the **Datasets** folder.

- **train_data.csv** – Used for training the machine learning models.
- **test_data.csv** – Used for testing and evaluating model performance.

Each record contains:

- News headline
- Label (FAKE or REAL)

---

## Output

After execution, the project generates:

- Classification Reports
- Confusion Matrix Images
- Accuracy Summary
- Misclassified Headlines
- Cross Validation Results
- Feature Importance Reports

All outputs are saved inside the **results/** folder.

---

## Machine Learning Models

- Multinomial Naive Bayes
- Decision Tree Classifier

---

## Future Enhancements

- Add a web interface using Flask or Streamlit.
- Support full news articles instead of only headlines.
- Integrate real-time news prediction.
- Improve accuracy using deep learning models such as LSTM or BERT.
- Deploy the project to the cloud.

---

## Author

Yasasri Amudalapalli
