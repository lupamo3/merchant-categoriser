import argparse, joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from prepare_data import load_and_split
from config import MODEL_DIR

def train(csv_path: str):
    X_train, X_test, y_train, y_test = load_and_split(csv_path)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1,2))),
        ("clf",   LogisticRegression(max_iter=1000, n_jobs=-1))
    ])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_DIR / "merchant_clf.joblib")
    print(f"✓ Saved model → {MODEL_DIR/'merchant_clf.joblib'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    train(**vars(parser.parse_args()))
