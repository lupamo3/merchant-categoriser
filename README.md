# 🛍️ Merchant‑Categoriser Toy Repo

A **plug‑and‑play sandbox** for experimenting with two text‑classification back‑ends:

1. **Scikit‑learn** – TF‑IDF + Logistic Regression  
2. **LLM API** – few‑shot prompting of OpenAI (or any Chat‑Completions‑compatible) models

The goal is to help you **compare speed, accuracy, and cost** on bank‑statement “narration → category” tasks before wiring a full production pipeline.

---

## 📂 Project layout


---

## 🚀 Quick‑start

```bash
# 1  Clone & enter
git clone https://github.com/<your-handle>/merchant-categoriser.git
cd merchant-categoriser

# 2  Create env & install deps
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3  Add data
#    Must contain columns:  description , category
cp /path/to/your/file.csv data/merchant_sample.csv
#    OR generate a tiny stub (500 rows):
python - <<'PY'
import csv, random, pathlib
categories = ["GROCERIES","DINING","ELECTRONICS","SALARY","TRANSPORT","UTILITIES"]
merchants = {"GROCERIES":["NAIVAS","CARREFOUR","TESCO"],
             "DINING":["MCDONALDS","KFC","DOMINOS"],
             "ELECTRONICS":["APPLE","AMAZON","JUMIA"],
             "SALARY":["PAYROLL ACME LTD"],
             "TRANSPORT":["UBER","BOLT","LYFT"],
             "UTILITIES":["KENYA POWER","WATER BOARD","SAFARICOM"]}
pathlib.Path("data").mkdir(exist_ok=True)
with open("data/merchant_sample.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["description","category"])
    for _ in range(500):
        cat=random.choice(categories)
        desc=f"{random.choice(merchants[cat])} POS {random.randint(1000,9999)}"
        w.writerow([desc,cat])
PY

# 4  Train the sklearn model
python src/train_sklearn.py --csv data/merchant_sample.csv

# 5  Evaluate
python src/evaluate.py --csv data/merchant_sample.csv --method sklearn
python src/evaluate.py --csv data/merchant_sample.csv --method llm \
       --openai-key $OPENAI_API_KEY

# 6  Predict one line
python src/predict.py --text "SAFARICOM MPESA 07XX123456" --method sklearn
