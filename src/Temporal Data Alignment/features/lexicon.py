import re
from collections import Counter

import pandas as pd

LM_CATEGORIES = ["Negative", "Positive", "Uncertainty", "Litigious"]
_TOKEN_RE = re.compile(r"[a-z]+")


# VADER
def vader_title_scores(titles: pd.Series) -> pd.Series:
    #Return the VADER compound score for every title
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    return titles.fillna("").astype(str).map(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )


# Loughran-McDonald
def load_lm_dictionary(csv_path: str) -> dict:
    lm = pd.read_csv(csv_path)
    lm["Word"] = lm["Word"].astype(str).str.lower()
    return {cat: set(lm.loc[lm[cat] > 0, "Word"]) for cat in LM_CATEGORIES}


def lm_body_scores(bodies: pd.Series, lm_dict: dict) -> pd.DataFrame:
    rows = []
    for text in bodies.fillna("").astype(str):
        tokens = _TOKEN_RE.findall(text.lower())
        counts = Counter(tokens)
        n_words = max(len(tokens), 1)
        row, raw = {}, {}
        for cat, words in lm_dict.items():
            hits = sum(c for w, c in counts.items() if w in words)
            raw[cat] = hits
            row[f"body_lm_{cat.lower()}"] = hits / n_words
        row["body_lm_net"] = (raw["Positive"] - raw["Negative"]) / (
            raw["Positive"] + raw["Negative"] + 1
        )
        row["body_word_count"] = len(tokens)
        rows.append(row)
    return pd.DataFrame(rows, index=bodies.index)
