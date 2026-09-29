"""
Turn article-level features into ONE ROW PER DAY.

Person 3 calls this AFTER mapping each article to its trading day, e.g.
    daily = aggregate_daily(articles, key="trade_date")
so weekend articles are combined into Monday's row, weighted per article.

Because Task 1 capped each sampled week at 100 articles, raw counts do not
reflect real news volume. Most features are therefore averages or shares;
gdelt_mentions_sum (from GDELT itself) is the better volume signal.
"""
import pandas as pd

NEG_THRESHOLD = -0.05   # standard VADER cut-off for "negative"


def aggregate_daily(df: pd.DataFrame, key: str = "trade_date") -> pd.DataFrame:
    df = df.copy()

    # helper columns must exist BEFORE grouping
    if "title_vader" in df:
        df["_title_is_neg"] = (df["title_vader"] < NEG_THRESHOLD).astype(int)
    if {"gdelt_tone", "gdelt_mentions"} <= set(df.columns):
        w = df["gdelt_mentions"].clip(lower=1)
        df["_tone_x_w"] = df["gdelt_tone"] * w
        df["_w"] = w

    g = df.groupby(key)
    out = pd.DataFrame({"news_count": g.size()})

    if "title_vader" in df:
        out["title_vader_mean"] = g["title_vader"].mean()
        out["title_vader_min"] = g["title_vader"].min()
        out["title_vader_std"] = g["title_vader"].std().fillna(0)
        out["title_neg_share"] = g["_title_is_neg"].mean()

    # mean of every body_ / ent_ / cameo column
    # (for 0/1 ent_ and cameo flags, the mean is the SHARE of articles)
    mean_cols = [c for c in df.columns
                 if c.startswith(("body_lm_", "body_svd_", "ent_", "gdelt_cameo_"))]
    if mean_cols:
        means = g[mean_cols].mean()
        means.columns = [c if c.startswith("body_") else f"{c}_share"
                         for c in mean_cols]
        out = out.join(means)

    if "gdelt_goldstein" in df:
        out["gdelt_goldstein_mean"] = g["gdelt_goldstein"].mean()
        out["gdelt_goldstein_min"] = g["gdelt_goldstein"].min()
    if "_w" in df:
        out["gdelt_tone_wmean"] = g["_tone_x_w"].sum() / g["_w"].sum()
        out["gdelt_mentions_sum"] = g["gdelt_mentions"].sum()

    return out.reset_index()
