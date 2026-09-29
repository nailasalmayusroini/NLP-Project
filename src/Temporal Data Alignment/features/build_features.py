"""
Build article-level NLP features from the Task 1 news output.

Two ways to run it:

1) From a notebook (e.g. Google Colab):
       from src.features.build_features import build
       build(articles="data/news.csv", lm="data/external/LM.csv",
             val_start="2025-01-01")

2) From a terminal, at the repo root:
       python -m src.features.build_features --articles data/news.csv \
           --lm data/external/LM.csv --val-start 2025-01-01

Outputs (paths relative to the project folder)
    data/processed/article_features.csv   one row per article, all features
    data/processed/daily_preview.csv      grouped by RAW news date (EDA / test
                                          merge only -- Person 3 re-aggregates
                                          by trade_date for modelling)
    data/processed/svd_top_terms.csv      top words per SVD theme
    models/tfidf_svd.joblib               TF-IDF + SVD fitted on TRAIN only
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from . import tfidf_svd
from .aggregate import aggregate_daily
from .entities import entity_flags
from .lexicon import lm_body_scores, load_lm_dictionary, vader_title_scores

# ---------------------------------------------------------------------------
# Column names. If you do not pass a column_map, the code GUESSES which of
# your columns is which using these candidate names (case, spaces and
# underscores are ignored). Only date, title and body are required.
# ---------------------------------------------------------------------------
COLUMN_CANDIDATES = {
    "date": ["date", "published_date", "publish_date", "pub_date",
             "published_at", "datetime", "timestamp", "sqldate", "dateadded"],
    "title": ["title", "headline", "article_title", "news_title"],
    "body": ["body", "text", "content", "full_text", "article_text",
             "clean_text", "cleaned_text", "article", "news_text"],
    "cameo_root": ["eventrootcode", "event_root_code", "cameo_root",
                   "root_code", "cameo"],
    "gdelt_goldstein": ["goldsteinscale", "goldstein_scale", "goldstein"],
    "gdelt_tone": ["avgtone", "avg_tone", "tone"],
    "gdelt_mentions": ["nummentions", "num_mentions", "mentions"],
}
REQUIRED = ["date", "title", "body"]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def guess_column_map(columns) -> dict:
    """Return {your_column_name: pipeline_name} for every column it recognises."""
    lookup = {_norm(c): c for c in columns}
    mapping = {}
    for target, candidates in COLUMN_CANDIDATES.items():
        for cand in candidates:
            if _norm(cand) in lookup and lookup[_norm(cand)] not in mapping:
                mapping[lookup[_norm(cand)]] = target
                break
    return mapping


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _parse_dates(s: pd.Series) -> pd.Series:
    """Handle YYYY-MM-DD strings, timestamps, and GDELT integer dates."""
    if pd.api.types.is_numeric_dtype(s):
        # GDELT SQLDATE (20240115) or DATEADDED (20240115123000)
        return pd.to_datetime(s.astype("Int64").astype(str).str[:8],
                              format="%Y%m%d", errors="coerce")
    d = pd.to_datetime(s, errors="coerce", format="mixed")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return d.dt.normalize()


def build(articles, lm, val_start, column_map=None, n_components=30,
          min_df=5, out_dir="data/processed", model_dir="models"):
    """Run the whole feature pipeline and save the outputs. Returns the
    article-level feature table."""
    out_dir, model_dir = Path(out_dir), Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. load Task 1 output ------------------------------------------------
    df = read_table(articles)
    if column_map is None:
        column_map = guess_column_map(df.columns)
    print("Column mapping used (your column -> pipeline name):")
    for k, v in column_map.items():
        print(f"   {k!r:>28} -> {v}")
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df})
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise KeyError(f"Could not find columns for {missing}. Your columns "
                       f"are {list(df.columns)}. Set COLUMN_MAP by hand.")

    df["date"] = _parse_dates(df["date"])
    bad = df["date"].isna().sum()
    if bad:
        print(f"WARNING: dropped {bad} rows with unreadable dates")
        df = df.dropna(subset=["date"])
    for col in ["gdelt_goldstein", "gdelt_tone", "gdelt_mentions"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    if "article_id" not in df:
        df.insert(0, "article_id", range(len(df)))
    print(f"Loaded {len(df):,} articles, {df['date'].min().date()} "
          f"to {df['date'].max().date()}")

    # 2. VADER on titles ---------------------------------------------------
    df["title_vader"] = vader_title_scores(df["title"])

    # 3. Loughran-McDonald on bodies ---------------------------------------
    df = df.join(lm_body_scores(df["body"], load_lm_dictionary(lm)))

    # 4. entity / theme keyword flags --------------------------------------
    df = df.join(entity_flags(df["title"], df["body"]))

    # 5. GDELT CAMEO root code -> one 0/1 column per code ------------------
    if "cameo_root" in df:
        codes = (pd.to_numeric(df["cameo_root"], errors="coerce")
                 .astype("Int64").astype(str).str.zfill(2))
        df = df.join(pd.get_dummies(codes, prefix="gdelt_cameo").astype(int))
        df = df.drop(columns=[c for c in df if c == "gdelt_cameo_<NA>"])

    # 6. TF-IDF + SVD, fitted on TRAIN period only -------------------------
    train_mask = df["date"] < pd.Timestamp(val_start)
    print(f"Fitting TF-IDF/SVD on {train_mask.sum():,} training articles "
          f"(before {val_start})")
    vec, svd = tfidf_svd.fit_tfidf_svd(df.loc[train_mask, "body"],
                                       n_components=n_components,
                                       min_df=min_df)
    df = df.join(tfidf_svd.transform_tfidf_svd(df["body"], vec, svd))
    tfidf_svd.save(vec, svd, model_dir / "tfidf_svd.joblib")
    tfidf_svd.top_terms_per_component(vec, svd).to_csv(
        out_dir / "svd_top_terms.csv", index=False)

    # 7. save ----------------------------------------------------------------
    feature_cols = [c for c in df.columns
                    if c.startswith(("title_", "body_", "ent_", "gdelt_"))]
    keep = ["article_id", "date", "title"] + feature_cols
    df[keep].to_csv(out_dir / "article_features.csv", index=False)
    aggregate_daily(df[keep], key="date").to_csv(
        out_dir / "daily_preview.csv", index=False)
    print(f"Saved {len(feature_cols)} feature columns to {out_dir}/ "
          f"and the TF-IDF/SVD model to {model_dir}/")
    return df[keep]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--articles", required=True)
    p.add_argument("--lm", required=True, help="LM Master Dictionary CSV")
    p.add_argument("--val-start", required=True,
                   help="first date of the validation period (from Person 3)")
    p.add_argument("--column-map", default=None,
                   help='JSON, e.g. \'{"content": "body"}\'; guessed if omitted')
    p.add_argument("--n-components", type=int, default=30)
    p.add_argument("--min-df", type=int, default=5)
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--model-dir", default="models")
    a = p.parse_args()
    build(a.articles, a.lm, a.val_start,
          column_map=json.loads(a.column_map) if a.column_map else None,
          n_components=a.n_components, min_df=a.min_df,
          out_dir=a.out_dir, model_dir=a.model_dir)


if __name__ == "__main__":
    main()
