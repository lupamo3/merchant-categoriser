import argparse, joblib
from llm_classifier import classify as llm_predict
from config import MODEL_DIR

model = None
def sklearn_predict(text):
    global model
    if model is None:
        model = joblib.load(MODEL_DIR / "merchant_clf.joblib")
    proba = model.predict_proba([text])[0]
    best = proba.argmax()
    return model.classes_[best], float(proba[best])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--method", choices=["sklearn", "llm"], default="sklearn")
    args = p.parse_args()

    cat, conf = (sklearn_predict if args.method=="sklearn" else llm_predict)(args.text)
    print(f"{args.method.upper()} ⇒ {cat}  (confidence {conf:.2f})")
