#!/usr/bin/env python3
# -------------------------------------------------------------
# model_test_fixed.py
#
# TESTING SCRIPT FOR FAKE vs REAL NEWS HEADLINE CLASSIFICATION
#
# What this script does:
# 1. Loads test dataset from Datasets/test_data.csv
# 2. Automatically cleans malformed rows if needed
# 3. Loads saved models + TF-IDF vectorizer from:
#       results/train_results/
# 4. Converts test headlines into TF-IDF vectors
# 5. Evaluates both trained models on unseen TEST data:
#       - Accuracy
#       - Precision
#       - Recall
#       - F1-Score
#       - Confusion Matrix
# 6. Saves classification reports and confusion matrix images
# 7. Saves misclassified headlines for both models
# 8. Creates summary_results_test.csv for final comparison
#
# Purpose:
# To measure real-world performance of both models on new data
# and compare which model generalizes better to unseen headlines.
# -------------------------------------------------------------

import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# ------------- Config -------------
TEST_FILENAME = "Datasets/test_data.csv"
TEST_CLEANED = "Datasets/test_data_cleaned.csv"
TRAIN_RESULTS_DIR = "results/train_results"
RESULTS_DIR = "results/test_results"
NB_MODEL_FILES = [os.path.join(TRAIN_RESULTS_DIR,"nb_model.pkl"), "nb_model.pkl"]
DT_MODEL_FILES = [os.path.join(TRAIN_RESULTS_DIR,"dt_model.pkl"), "dt_model.pkl"]
VECTORIZER_FILES = [os.path.join(TRAIN_RESULTS_DIR,"tfidf.pkl"), "tfidf.pkl"]
LABELS = ["FAKE","REAL"]

# ------------- Utilities -------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f: f.write(text)
    print(f"Saved text file: {path}")

def save_confusion_matrix(cm, classes, outpath, title):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           ylabel="True label", xlabel="Predicted label", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max()/2.0 if cm.max() != 0 else 0.5
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j,i, format(int(cm[i,j]), "d"), ha="center", va="center", color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150); plt.close(fig)
    print(f"Saved confusion matrix: {outpath}")

# ------------- Load test CSV robustly -------------
def load_test_csv(primary=TEST_FILENAME, fallback=TEST_CLEANED):
    if os.path.exists(primary): fname = primary
    elif os.path.exists(fallback): fname = fallback
    else:
        print(f"ERROR: test file not found. Checked: {primary}, {fallback}"); sys.exit(1)
    print(f"Loading TEST dataset from file: {fname}")
    try:
        df = pd.read_csv(fname, engine='python', on_bad_lines='skip')
    except TypeError:
        df = pd.read_csv(fname, engine='python', error_bad_lines=False, warn_bad_lines=True)  # type: ignore
    except Exception as e:
        print("Error reading test CSV:", e); sys.exit(1)
    df.columns = [c.strip() for c in df.columns]
    if "title" not in df.columns or "label" not in df.columns:
        # attempt to recover from single combined column
        possible = [c for c in df.columns if "," in c]
        if possible:
            print("Attempting to reparse combined column...")
            raw = df[possible[0]].astype(str).tolist()
            import io
            try:
                df2 = pd.read_csv(io.StringIO("\n".join(raw)), engine='python', on_bad_lines='skip')
                df = df2; df.columns = [c.strip() for c in df.columns]; print("Recovered dataset.")
            except Exception as e:
                print("Recovery failed:", e)
    if "title" not in df.columns or "label" not in df.columns:
        print("ERROR: TEST dataset must contain 'title' and 'label' columns. Found:", list(df.columns)); sys.exit(1)
    df['label'] = df['label'].astype(str).str.strip().str.upper()
    df = df[['title','label']].dropna().reset_index(drop=True)
    # keep only FAKE/REAL
    before = len(df); df = df[df['label'].isin(['FAKE','REAL'])].reset_index(drop=True); removed = before - len(df)
    if removed>0: print(f"Removed {removed} rows with invalid labels from TEST.")
    return df

# ------------- Evaluate helper -------------
def evaluate_model(model, X, y, labels=LABELS):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='macro', zero_division=0)
    report = classification_report(y, y_pred, digits=4)
    cm = confusion_matrix(y, y_pred, labels=labels)
    return {"accuracy":acc, "precision_macro":precision, "recall_macro":recall, "f1_macro":f1, "report":report, "confusion_matrix":cm, "y_pred":y_pred}

# ------------- Main -------------
def main():
    ensure_dir(RESULTS_DIR)
    df_test = load_test_csv()
    print(f"TEST dataset rows: {len(df_test)}  Label counts:\n{df_test['label'].value_counts()}\n")

    # 1) load models/vectorizer - try train_results first
    nb = dt = vectorizer = None
    for p in NB_MODEL_FILES:
        if os.path.exists(p):
            try:
                nb = joblib.load(p); print("Loaded NB model from", p); break
            except Exception as e:
                print("Failed loading NB from", p, "->", e)
    for p in DT_MODEL_FILES:
        if os.path.exists(p):
            try:
                dt = joblib.load(p); print("Loaded DT model from", p); break
            except Exception as e:
                print("Failed loading DT from", p, "->", e)
    for p in VECTORIZER_FILES:
        if os.path.exists(p):
            try:
                vectorizer = joblib.load(p); print("Loaded vectorizer from", p); break
            except Exception as e:
                print("Failed loading vectorizer from", p, "->", e)
    if nb is None or dt is None or vectorizer is None:
        print("ERROR: could not load all required model/vectorizer files. Checked:\n", NB_MODEL_FILES, DT_MODEL_FILES, VECTORIZER_FILES)
        sys.exit(1)

    X_test_text = df_test['title'].values; y_test = df_test['label'].values
    try:
        X_test = vectorizer.transform(X_test_text)
    except Exception as e:
        print("Vectorizer transform failed:", e); sys.exit(1)

    print("Evaluating Naive Bayes on TEST dataset...")
    res_nb = evaluate_model(nb, X_test, y_test, labels=LABELS)
    print("Evaluating Decision Tree on TEST dataset...")
    res_dt = evaluate_model(dt, X_test, y_test, labels=LABELS)

    # Save confusion matrices and reports
    ensure_dir(RESULTS_DIR)
    save_confusion_matrix(res_nb['confusion_matrix'], LABELS, os.path.join(RESULTS_DIR, "cm_naive_bayes_test.png"), "Naive Bayes Confusion Matrix (TEST)")
    save_confusion_matrix(res_dt['confusion_matrix'], LABELS, os.path.join(RESULTS_DIR, "cm_decision_tree_test.png"), "Decision Tree Confusion Matrix (TEST)")

    save_text(res_nb['report'], os.path.join(RESULTS_DIR, "classification_report_naive_bayes_test.txt"))
    save_text(res_dt['report'], os.path.join(RESULTS_DIR, "classification_report_decision_tree_test.txt"))

    summary = pd.DataFrame([
        {"model":"MultinomialNB_TEST","accuracy":res_nb['accuracy'],"precision_macro":res_nb['precision_macro'],"recall_macro":res_nb['recall_macro'],"f1_macro":res_nb['f1_macro']},
        {"model":"DecisionTree_TEST","accuracy":res_dt['accuracy'],"precision_macro":res_dt['precision_macro'],"recall_macro":res_dt['recall_macro'],"f1_macro":res_dt['f1_macro']}
    ])
    summary.to_csv(os.path.join(RESULTS_DIR,"summary_results_test.csv"), index=False)
    print("Saved TEST summary CSV.")

    # Misclassified examples
    df_err_nb = pd.DataFrame({"title":X_test_text,"true":y_test,"pred":res_nb["y_pred"]})
    df_err_nb = df_err_nb[df_err_nb["true"] != df_err_nb["pred"]]
    df_err_nb.to_csv(os.path.join(RESULTS_DIR,"misclassified_nb_test.csv"), index=False)
    print(f"Saved misclassified_nb_test.csv (count: {len(df_err_nb)})")

    df_err_dt = pd.DataFrame({"title":X_test_text,"true":y_test,"pred":res_dt["y_pred"]})
    df_err_dt = df_err_dt[df_err_dt["true"] != df_err_dt["pred"]]
    df_err_dt.to_csv(os.path.join(RESULTS_DIR,"misclassified_dt_test.csv"), index=False)
    print(f"Saved misclassified_dt_test.csv (count: {len(df_err_dt)})")

    print("\n======== FINAL SUMMARY (TEST DATA) ========")
    print(summary)
    best = summary.loc[summary["f1_macro"].idxmax()]
    print(f"On TEST dataset, best model (by macro-F1) = {best['model']} (f1_macro={best['f1_macro']:.4f})")
    print(f"All TEST results saved in '{RESULTS_DIR}/'.")

if __name__ == "__main__":
    main()
