import pandas as pd, re, unicodedata
from sklearn.model_selection import train_test_split

def clean(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def load_and_split(csv_path: str, test_size=0.2, seed=42):
    df = pd.read_csv(csv_path)
    df["clean_desc"] = df["description"].apply(clean)
    return train_test_split(df["clean_desc"], df["category"],
                            test_size=test_size, random_state=seed)
