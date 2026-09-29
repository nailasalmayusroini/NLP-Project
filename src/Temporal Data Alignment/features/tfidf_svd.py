"""
TF-IDF + SVD "theme" features for article bodies.

TF-IDF scores how important each word is in an article (frequent here, rare
elsewhere). That gives thousands of columns, so TruncatedSVD compresses them
into a few dozen "themes".

LEAKAGE RULE: fit ONLY on training-period articles, save the fitted objects,
and reuse them to transform validation/test articles. (In Task 1, TF-IDF for
dedup could be fit on everything; for prediction features it cannot.)
"""
import joblib
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf_svd(train_bodies: pd.Series, n_components: int = 30,
                  min_df: int = 5, max_features: int = 20000,
                  random_state: int = 42):
    """Fit TF-IDF and SVD on TRAINING bodies only. Returns (vectorizer, svd)."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",   # only real words: drops numbers and codes like "e96"
        stop_words="english",
        min_df=min_df,        # ignore words in fewer than min_df articles
        max_df=0.8,           # ignore words in more than 80% of articles
        max_features=max_features,
        sublinear_tf=True,    # dampen very repeated words
    )
    X = vectorizer.fit_transform(train_bodies.fillna("").astype(str))
    n_components = min(n_components, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd.fit(X)
    return vectorizer, svd


def transform_tfidf_svd(bodies: pd.Series, vectorizer, svd) -> pd.DataFrame:
    """Apply the saved vectorizer + SVD to any set of bodies."""
    X = vectorizer.transform(bodies.fillna("").astype(str))
    Z = svd.transform(X)
    cols = [f"body_svd_{i}" for i in range(Z.shape[1])]
    return pd.DataFrame(Z, columns=cols, index=bodies.index)


def top_terms_per_component(vectorizer, svd, n_terms: int = 10) -> pd.DataFrame:
    """Top words for each SVD theme -- handy for the EDA notebook and report."""
    terms = vectorizer.get_feature_names_out()
    rows = []
    for i, comp in enumerate(svd.components_):
        top = comp.argsort()[::-1][:n_terms]
        rows.append({"component": f"body_svd_{i}",
                     "top_terms": ", ".join(terms[j] for j in top)})
    return pd.DataFrame(rows)


def save(vectorizer, svd, path: str):
    joblib.dump({"vectorizer": vectorizer, "svd": svd}, path)


def load(path: str):
    obj = joblib.load(path)
    return obj["vectorizer"], obj["svd"]
