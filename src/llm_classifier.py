import os, json, openai
from config import CATEGORIES

# call once at import
openai.api_key = os.getenv("OPENAI_API_KEY")

_SYSTEM = (
    "You are an API that classifies a bank-statement line into one of the "
    f"categories: {', '.join(CATEGORIES)}. "
    "Reply with JSON: {\"category\": <CATEGORY>, \"confidence\": <0-1>}"
)

def classify(text: str) -> tuple[str, float]:
    msg = f"Description: \"{text}\""
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": msg}],
        temperature=0.0,
        max_tokens=20,
    )
    payload = json.loads(rsp.choices[0].message.content)
    return payload["category"], payload["confidence"]
