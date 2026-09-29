import re

import pandas as pd

ENTITY_KEYWORDS = {
    # actors
    "us": ["united states", "u.s.", "washington", "white house", "biden",
           "trump", "american"],
    "fed": ["federal reserve", "the fed", "fomc", "powell", "rate hike",
            "rate cut"],
    "china": ["china", "chinese", "beijing", "xi jinping"],
    "indonesia": ["indonesia", "indonesian", "jakarta", "bank indonesia",
                  "rupiah", "prabowo", "jokowi"],
    "russia": ["russia", "russian", "moscow", "kremlin", "putin"],
    "ukraine": ["ukraine", "ukrainian", "kyiv", "zelensky"],
    "middle_east": ["israel", "iran", "gaza", "saudi", "middle east",
                    "houthi", "red sea"],
    # themes
    "sanctions": ["sanction", "embargo", "export control"],
    "tariffs": ["tariff", "trade war", "import duty", "import duties"],
    "conflict": ["war", "attack", "missile", "invasion", "military",
                 "troops", "airstrike"],
    "election": ["election", "ballot", "vote"],
    "oil": ["oil", "crude", "opec", "brent"],
    "inflation": ["inflation", "consumer prices", "cpi"],
}


def _compile(words):
    alt = "|".join(re.escape(w) for w in words)
    return re.compile(rf"(?<![a-z])(?:{alt})s?(?![a-z])")


_PATTERNS = {name: _compile(words) for name, words in ENTITY_KEYWORDS.items()}

_US_CAPS = re.compile(r"(?<![A-Za-z])US(?![A-Za-z])")


def entity_flags(titles: pd.Series, bodies: pd.Series) -> pd.DataFrame:
    """Return one 0/1 column per keyword group, named ent_<group>."""
    raw = titles.fillna("").astype(str) + " " + bodies.fillna("").astype(str)
    text = raw.str.lower()
    out = {f"ent_{name}": text.str.contains(pat).astype(int)
           for name, pat in _PATTERNS.items()}
    out["ent_us"] = (out["ent_us"] | raw.str.contains(_US_CAPS)).astype(int)
    return pd.DataFrame(out, index=titles.index)
