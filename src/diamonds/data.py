"""Loading and cleaning the diamonds dataset."""

from pathlib import Path

import pandas as pd

NUMERIC_COLUMNS = ["carat", "depth", "table", "price", "x", "y", "z"]
CATEGORICAL_COLUMNS = ["cut", "color", "clarity"]
REQUIRED_COLUMNS = ["carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"]
DIMENSION_COLUMNS = ["x", "y", "z"]

# Remove likely measurement errors above 20 mm.
MAX_DIMENSION_MM = 20.0

# Grades run from worst to best.
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

ORDINAL_LEVELS = {
    "cut": CUT_ORDER,
    "color": COLOR_ORDER,
    "clarity": CLARITY_ORDER,
}


def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load a local CSV, or the seaborn dataset if no path is given."""
    if csv_path is not None and not isinstance(csv_path, (str, Path)):
        raise TypeError("csv_path must be a string, Path, or None")

    if csv_path is None:
        import seaborn as sns

        df = sns.load_dataset("diamonds")
    else:
        df = pd.read_csv(csv_path)

    missing = []

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    # loading can produce diff dataType
    for col in CATEGORICAL_COLUMNS: 
        df[col] = df[col].astype(str)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col])
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove duplicates and invalid dimensions, keeping counts for each rule."""
    removed = {}

    before = len(df)
    df = df.drop_duplicates()
    removed["duplicates"] = before - len(df)

    before = len(df)
    df = df[(df[DIMENSION_COLUMNS] > 0).all(axis=1)]
    removed["zero_dimensions"] = before - len(df)

    before = len(df)
    df = df[(df[DIMENSION_COLUMNS] <= MAX_DIMENSION_MM).all(axis=1)]
    removed["outliers"] = before - len(df)

    return df.reset_index(drop=True), removed


def encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Encode cut, color and clarity from worst to best, leaving the input unchanged."""
    out = df.copy()
    for col, levels in ORDINAL_LEVELS.items():
        codes = {level: i for i, level in enumerate(levels)}
        unknown = sorted(set(out[col].astype(str)) - codes.keys())
        if unknown:
            raise ValueError(f"Unknown {col} value(s) {unknown}; expected one of {levels}")
        out[col] = out[col].astype(str).map(codes).astype("int64")
    return out
