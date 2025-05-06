import argparse, pandas as pd, tqdm, numpy as np
from sklearn.metrics import classification_report
from prepare_data import clean
from predict import sklearn_predict, llm_predict

def evaluate(csv_path, method):
    df = pd.read_csv(csv_path)
    preds, confs = [], []
    fn = sklearn_predict if method=="sklearn" else llm_predict
    for desc in tqdm.tqdm(df["description"]):
        cat, conf = fn(desc)
        preds.append(cat); confs.append(conf)
    print(classification_report(df["category"], preds, digits=3))
    print(f"Avg confidence: {np.mean(confs):.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--method", choices=["sklearn", "llm"], default="sklearn")
    evaluate(**vars(ap.parse_args()))
