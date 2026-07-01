#!/usr/bin/env python3
# -------------------------------------------------------------
# model_train_fixed.py
#
# TRAINING SCRIPT FOR FAKE vs REAL NEWS HEADLINE CLASSIFICATION
#
# What this script does:
# 1. Loads the training dataset from Datasets/train_data.csv
# 2. Automatically fixes malformed CSV rows (extra commas / broken text)
# 3. Cleans labels and keeps only FAKE and REAL
# 4. Converts headlines into TF-IDF numeric features
# 5. Splits into training/testing sets (with safe fallback)
# 6. Trains two ML models:
#       - Multinomial Naive Bayes
#       - Decision Tree (entropy, ID3-style)
# 7. Performs 5-fold Cross-Validation (Accuracy, Precision, Recall, F1)
# 8. Evaluates both models on the hold-out test split
# 9. Generates:
#       - Confusion matrices
#       - Classification reports
#       - Misclassified headline CSV files
#       - Top feature words (NB + Decision Tree)
#       - Summary results table
# 10. Saves trained models and vectorizer inside:
#       results/train_results/ (nb_model.pkl, dt_model.pkl, tfidf.pkl)
#
# Purpose:
# To train, compare, and analyze both models according to syllabus metrics:
# Accuracy, Precision, Recall, F1-Score, Confusion Matrix,
# and Overfitting/Underfitting discussion.
# -------------------------------------------------------------

import os
import sys
import csv
import traceback
import itertools
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import make_pipeline

# ------------- Config -------------
RANDOM_STATE = 42
LOCAL_FILENAME = "Datasets/train_data.csv"
CLEANED_FILENAME = "Datasets/train_data_cleaned.csv"
RESULTS_DIR = "results/train_results"
MAX_FEATURES = 5000
TEST_SIZE = 0.30
EXPECTED_COLUMNS = 4
AUTO_SALVAGE = True
ON_BAD_LINES = "skip"
CV_FOLDS = 5

# ------------- Utilities -------------
def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def save_text_file(text, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved text file: {path}")

def save_list_file(lines, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Saved list file: {path}")

def save_confusion_matrix(cm, classes, outname, title):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        ylabel="True label", xlabel="Predicted label", title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0 if cm.max() != 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, outname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix: {path}")

# ------------- CSV diagnostics + salvage -------------
def find_bad_rows_csv(filename, expected_columns=EXPECTED_COLUMNS):
    bad = []
    try:
        with open(filename, newline='', encoding='utf-8', errors='replace') as fin:
            reader = csv.reader(fin)
            for i, row in enumerate(reader, start=1):
                if len(row) != expected_columns:
                    bad.append((i, len(row), row))
    except Exception as e:
        print("Error scanning CSV:", e)
    print(f"Scanned {filename}. Found {len(bad)} bad rows (expected {expected_columns} columns).")
    return bad

def salvage_and_write(filename, outname, expected_columns=EXPECTED_COLUMNS):
    kept = salvaged = dropped = 0
    try:
        with open(filename, newline='', encoding='utf-8', errors='replace') as fin, \
             open(outname, 'w', newline='', encoding='utf-8') as fout:
            reader = csv.reader(fin)
            writer = csv.writer(fout)
            for i, row in enumerate(reader, start=1):
                if len(row) == expected_columns:
                    writer.writerow(row); kept += 1
                elif len(row) > expected_columns:
                    # Join extras into the last column (safe)
                    new_row = row[: expected_columns - 1] + [','.join(row[expected_columns - 1:])]
                    writer.writerow(new_row); salvaged += 1
                else:
                    # fewer columns - drop
                    dropped += 1
        print(f"Salvage complete. kept={kept} salvaged={salvaged} dropped={dropped}. Saved: {outname}")
    except Exception as e:
        print("Failed to salvage:", e)
        traceback.print_exc()
    return kept, salvaged, dropped

# ------------- Load dataset robustly -------------
def load_dataset_local_only(filename=LOCAL_FILENAME):
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found.")
        sys.exit(1)
    print(f"Loading dataset from local file: {filename}")
    try:
        df = pd.read_csv(filename)
        print("Loaded with pandas.read_csv() (default).")
    except pd.errors.ParserError as pe:
        print("ParserError:", pe)
        bad = find_bad_rows_csv(filename)
        if bad and AUTO_SALVAGE:
            print("Attempting salvage...")
            salvage_and_write(filename, CLEANED_FILENAME)
            try:
                df = pd.read_csv(CLEANED_FILENAME)
                print(f"Loaded cleaned CSV: {CLEANED_FILENAME}")
            except Exception as e:
                print("Retry failed, trying engine='python', on_bad_lines='skip' fallback.")
                df = pd.read_csv(filename, engine='python', on_bad_lines=ON_BAD_LINES)
        else:
            df = pd.read_csv(filename, engine='python', on_bad_lines=ON_BAD_LINES)
    except Exception as e:
        print("Unexpected read error:", e)
        df = pd.read_csv(filename, engine='python', on_bad_lines=ON_BAD_LINES)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Normalize labels to uppercase strings and strip
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.upper()

    # Validate presence of 'title' and 'label'
    if "title" not in df.columns or "label" not in df.columns:
        print("ERROR: dataset must contain 'title' and 'label' columns. Found:", list(df.columns))
        sys.exit(1)

    # Keep only rows where label is FAKE or REAL (defensive cleaning)
    df = df[["title", "label"]].dropna().reset_index(drop=True)
    before = len(df)
    df = df[df["label"].isin(["FAKE", "REAL"])].reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed} rows with invalid labels (kept only FAKE/REAL).")

    if len(df) == 0:
        print("ERROR: no usable rows after label cleaning.")
        sys.exit(1)
    return df

# ------------- Preprocess + TF-IDF -------------
def preprocess_vectorize(df, test_size=TEST_SIZE, max_features=MAX_FEATURES, random_state=RANDOM_STATE):
    X = df["title"].values
    y = df["label"].values

    # Check min class count for stratify safety
    vc = pd.Series(y).value_counts()
    min_count = vc.min()
    stratify_arg = y if min_count >= 2 else None
    if stratify_arg is None:
        print("WARNING: At least one class has <2 members; doing non-stratified split.")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_arg, random_state=random_state
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    return X_train, X_test, y_train, y_test, vectorizer, X_train_text, X_test_text

# ------------- Train helpers -------------
def train_nb(X_train, y_train):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    return nb

def train_dt(X_train, y_train, max_depth=20):
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, class_weight="balanced", random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    return dt

def evaluate(model, X_test, y_test, labels=["FAKE", "REAL"]):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return {"accuracy":acc, "precision_macro":precision, "recall_macro":recall, "f1_macro":f1, "report":report, "confusion_matrix":cm, "y_pred":y_pred}

def get_nb_top_features(nb_model, vectorizer, top_n=30):
    feature_names = vectorizer.get_feature_names_out()
    classes = list(nb_model.classes_)
    try:
        i_fake = classes.index("FAKE"); i_real = classes.index("REAL")
    except ValueError:
        i_fake, i_real = 0, 1
    logp = nb_model.feature_log_prob_
    diff = logp[i_fake] - logp[i_real]
    top_fake_idx = np.argsort(diff)[-top_n:][::-1]; top_real_idx = np.argsort(diff)[:top_n]
    top_fake = [(feature_names[i], float(diff[i])) for i in top_fake_idx]
    top_real = [(feature_names[i], float(-diff[i])) for i in top_real_idx]
    return top_fake, top_real

def save_nb_top_features(nb_model, vectorizer, top_n=30):
    top_fake, top_real = get_nb_top_features(nb_model, vectorizer, top_n=top_n)
    lines = ["Top words that predict FAKE (word, log-odds diff):"]
    lines += [f"{w}\t{score:.6f}" for w, score in top_fake]
    lines += ["", "Top words that predict REAL (word, log-odds diff):"]
    lines += [f"{w}\t{score:.6f}" for w, score in top_real]
    save_list_file(lines, "top_features_nb.txt")

def save_dt_top_features(dt_model, vectorizer, top_n=30):
    if not hasattr(dt_model, "feature_importances_"):
        save_list_file(["Decision Tree has no feature_importances_."], "top_features_dt.txt"); return
    imp = dt_model.feature_importances_
    if imp.sum() == 0:
        save_list_file(["Decision Tree feature importances are all zero."], "top_features_dt.txt"); return
    feature_names = vectorizer.get_feature_names_out()
    idx = np.argsort(imp)[::-1][:top_n]
    lines = ["Top Decision Tree features (word, importance):"] + [f"{feature_names[i]}\t{imp[i]:.8f}" for i in idx]
    save_list_file(lines, "top_features_dt.txt")

def save_misclassified_examples(X_test_text, y_test, y_pred, filename):
    df_err = pd.DataFrame({"title": X_test_text, "true": y_test, "pred": y_pred})
    df_err = df_err[df_err["true"] != df_err["pred"]]
    path = os.path.join(RESULTS_DIR, filename)
    df_err.to_csv(path, index=False)
    print(f"Saved misclassified examples: {path} (count: {len(df_err)})")

def run_cross_validation(df, folds=CV_FOLDS):
    X = df["title"].values; y = df["label"].values
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    pipe_nb = make_pipeline(TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES), MultinomialNB())
    pipe_dt = make_pipeline(TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES),
                            DecisionTreeClassifier(criterion="entropy", max_depth=20, class_weight="balanced", random_state=RANDOM_STATE))
    scoring = {"accuracy":"accuracy","precision_macro":"precision_macro","recall_macro":"recall_macro","f1_macro":"f1_macro"}
    results = []
    for name, pipe in [("MultinomialNB", pipe_nb), ("DecisionTree", pipe_dt)]:
        print(f"Running {folds}-fold CV for {name} ...")
        row = {"model":name}
        for sc_name, sc in scoring.items():
            try:
                scores = cross_val_score(pipe, X, y, cv=skf, scoring=sc, n_jobs=-1)
            except Exception as e:
                print("CV scoring error (fallback to single-thread, smaller folds):", e)
                scores = cross_val_score(pipe, X, y, cv=min(3, folds), scoring=sc, n_jobs=1)
            row[f"{sc_name}_mean"] = float(np.mean(scores)); row[f"{sc_name}_std"] = float(np.std(scores))
            print(f"  {sc_name}: mean={row[f'{sc_name}_mean']:.4f} std={row[f'{sc_name}_std']:.4f}")
        results.append(row)
    df_cv = pd.DataFrame(results)
    df_cv.to_csv(os.path.join(RESULTS_DIR, "cv_summary.csv"), index=False)
    print("Saved CV summary.")
    return df_cv

# ------------- Main -------------
def main():
    ensure_results_dir()
    print("Loading training dataset...")
    df = load_dataset_local_only()
    print(f"Dataset rows: {len(df)}  Label counts:\n{df['label'].value_counts()}\n")

    print("Running stratified cross-validation (pipelines) ...")
    try:
        df_cv = run_cross_validation(df, folds=CV_FOLDS)
    except Exception as e:
        print("CV failed:", e); df_cv = None

    print("Preprocessing and vectorizing for hold-out split...")
    X_train, X_test, y_train, y_test, vectorizer, X_train_text, X_test_text = preprocess_vectorize(df)

    # Train models
    print("Training Multinomial Naive Bayes...")
    nb = train_nb(X_train, y_train)
    print("Training Decision Tree...")
    dt = train_dt(X_train, y_train, max_depth=20)

    # Evaluate
    print("Evaluating Naive Bayes on hold-out split...")
    res_nb = evaluate(nb, X_test, y_test)
    print("Evaluating Decision Tree on hold-out split...")
    res_dt = evaluate(dt, X_test, y_test)

    labels = ["FAKE", "REAL"]
    save_confusion_matrix(res_nb["confusion_matrix"], labels, "cm_naive_bayes.png", "Naive Bayes Confusion Matrix (Train Split)")
    save_confusion_matrix(res_dt["confusion_matrix"], labels, "cm_decision_tree.png", "Decision Tree Confusion Matrix (Train Split)")

    save_nb_top_features(nb, vectorizer, top_n=30)
    save_dt_top_features(dt, vectorizer, top_n=30)

    save_misclassified_examples(X_test_text, y_test, res_nb["y_pred"], "misclassified_nb.csv")
    save_misclassified_examples(X_test_text, y_test, res_dt["y_pred"], "misclassified_dt.csv")

    summary = pd.DataFrame([
        {"model":"MultinomialNB","accuracy":res_nb["accuracy"],"precision_macro":res_nb["precision_macro"],"recall_macro":res_nb["recall_macro"],"f1_macro":res_nb["f1_macro"]},
        {"model":"DecisionTree","accuracy":res_dt["accuracy"],"precision_macro":res_dt["precision_macro"],"recall_macro":res_dt["recall_macro"],"f1_macro":res_dt["f1_macro"]}
    ])
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_results.csv"), index=False)
    print("Saved summary CSV.")

    # Reports
    report_lines = []
    report_lines.append("TRAIN REPORT")
    report_lines.append(f"Total rows: {len(df)}")
    report_lines.append(str(df['label'].value_counts()))
    report_lines.append("\nNaive Bayes report:\n"); report_lines.append(res_nb["report"])
    report_lines.append("\nDecision Tree report:\n"); report_lines.append(res_dt["report"])
    save_text_file("\n".join(report_lines), "report.txt")

    # Save classification reports
    save_text_file(res_nb["report"], "classification_report_naive_bayes.txt")
    save_text_file(res_dt["report"], "classification_report_decision_tree.txt")

    # Save models & vectorizer into results folder (so model_test can find them)
    nb_path = os.path.join(RESULTS_DIR, "nb_model.pkl")
    dt_path = os.path.join(RESULTS_DIR, "dt_model.pkl")
    tfidf_path = os.path.join(RESULTS_DIR, "tfidf.pkl")
    joblib.dump(nb, nb_path)
    joblib.dump(dt, dt_path)
    joblib.dump(vectorizer, tfidf_path)
    print(f"Saved models/vectorizer to {RESULTS_DIR}")

    print("\nFINAL SUMMARY (train split):")
    print(summary)
    best = summary.loc[summary["f1_macro"].idxmax()]
    print(f"Recommendation (by macro-F1): Best model = {best['model']} (f1_macro={best['f1_macro']:.4f})")
    if df_cv is not None:
        print("\nCV summary (saved).")

if __name__ == "__main__":
    main()
