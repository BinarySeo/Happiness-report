"""
happiness.py
------------
Shared data loading, statistics and charts for the World Happiness Report 2021 notebooks.

Keeping this logic in one tested module (instead of copy-pasted notebook cells) is what
prevents mix-ups like plotting one region's average under another region's label, or
drawing one factor's trend line on another factor's chart.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "DataFile" / "world-happiness-report-2021.csv"
IMG_DIR = ROOT / "img"

SCORE = "Ladder score"
REGION = "Regional indicator"

# The six factors the report uses to explain the ladder score → short labels for charts
FACTORS = {
    "Logged GDP per capita": "GDP",
    "Social support": "Support",
    "Healthy life expectancy": "Life Expectancy",
    "Freedom to make life choices": "Freedom",
    "Generosity": "Generosity",
    "Perceptions of corruption": "Perception of Corruption",
}

# Broader regional groups used for the regional heatmaps. Each of the report's
# 10 regional indicators belongs to exactly one group.
REGION_GROUPS = {
    "Asia": ["East Asia", "Southeast Asia", "South Asia"],
    "Europe": ["Western Europe", "Central and Eastern Europe"],
    "Americas, Australia & NZ": ["Latin America and Caribbean", "North America and ANZ"],
    "Africa & Middle East": ["Sub-Saharan Africa", "Middle East and North Africa"],
    "Commonwealth of Independent States": ["Commonwealth of Independent States"],
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the 2021 report, keeping country, region, score and the six raw factors."""
    df = pd.read_csv(path)
    columns = ["Country name", REGION, SCORE, *FACTORS]
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected file format, missing columns: {sorted(missing)}")
    return df[columns].sort_values(SCORE, ascending=False).reset_index(drop=True)


def numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Score + factor columns only (what correlation matrices should use)."""
    return df[[SCORE, *FACTORS]]


def region_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Average ladder score and country count per regional indicator, highest first."""
    return (
        df.groupby(REGION)[SCORE]
        .agg(avg_score="mean", countries="count")
        .sort_values("avg_score", ascending=False)
        .reset_index()
    )


def factor_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson r and p-value of each factor against the ladder score, strongest first."""
    rows = []
    for column, label in FACTORS.items():
        result = stats.pearsonr(df[column], df[SCORE])
        rows.append({"factor": label, "column": column,
                     "correlation": result.statistic, "p_value": result.pvalue})
    out = pd.DataFrame(rows)
    return out.reindex(out["correlation"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def top_and_bottom(df: pd.DataFrame, n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The n happiest and n least happy countries by ladder score."""
    return df.nlargest(n, SCORE), df.nsmallest(n, SCORE)


def region_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Countries in one of REGION_GROUPS."""
    return df[df[REGION].isin(REGION_GROUPS[group])]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def save(fig: plt.Figure, filename: str) -> Path:
    """Save a chart into img/ (used for the README)."""
    IMG_DIR.mkdir(exist_ok=True)
    path = IMG_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def region_bar_chart(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of average ladder score by region."""
    scores = region_scores(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=scores, x="avg_score", y=REGION, hue=REGION, legend=False, ax=ax)
    for value, y in zip(scores["avg_score"], range(len(scores))):
        ax.text(value + 0.05, y, f"{value:.2f}", va="center")
    ax.set(title="Average Ladder Score by Region (2021)", xlabel="Average ladder score", ylabel="")
    ax.set_xlim(0, scores["avg_score"].max() + 0.8)
    return fig


def factor_scatter(df: pd.DataFrame, column: str) -> plt.Figure:
    """Scatter of one factor vs. ladder score, colored by region, with that factor's own trend line."""
    clean = df[[column, SCORE, REGION]].dropna()
    r = stats.pearsonr(clean[column], clean[SCORE]).statistic
    slope, intercept = np.polyfit(clean[column], clean[SCORE], 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=clean, x=column, y=SCORE, hue=REGION, ax=ax)
    xs = np.linspace(clean[column].min(), clean[column].max(), 100)
    ax.plot(xs, slope * xs + intercept, color="m", label=f"trend (r = {r:.2f})")
    ax.set_title(f"{FACTORS.get(column, column)} vs Ladder Score")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    return fig


def corr_heatmap(df: pd.DataFrame, title: str) -> plt.Figure:
    """Lower-triangle correlation heatmap of score and factors."""
    corr = numeric(df).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0,
                cmap="PiYG", square=True, ax=ax)
    ax.set_title(f"{title} (n = {len(df)})")
    return fig
