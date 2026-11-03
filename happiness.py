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


# ---------------------------------------------------------------------------
# Combined model: which factors matter once the others are held constant?
# Plain NumPy/SciPy OLS so no extra dependencies are needed.
# ---------------------------------------------------------------------------

def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least squares with intercept → (coefficients incl. intercept, residuals, R²)."""
    A = np.column_stack([np.ones(len(y)), X])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ coef
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return coef, resid, r2


def _standardize(values: pd.DataFrame | pd.Series):
    return (values - values.mean()) / values.std(ddof=1)


def regression_summary(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """
    Multiple regression of ladder score on all six factors, in standardized units
    (beta = change in score, in standard deviations, per 1 SD change in the factor
    with the other five held constant).

    Returns (per-factor table, model stats). The table has:
        beta, p_value          classic OLS t-test
        ci_low, ci_high        95% bootstrap interval (resampling countries)
        unique_r2              R² lost when the factor is dropped = variance only it explains
        vif                    variance inflation factor (overlap with the other factors)
    """
    columns = list(FACTORS)
    X = _standardize(df[columns]).to_numpy()
    y = _standardize(df[SCORE]).to_numpy()
    n, k = X.shape

    coef, resid, r2 = _ols(X, y)
    A = np.column_stack([np.ones(n), X])
    dof = n - k - 1
    se = np.sqrt(np.diag(resid @ resid / dof * np.linalg.inv(A.T @ A)))
    p_values = 2 * stats.t.sf(np.abs(coef / se), dof)

    rng = np.random.default_rng(seed)
    boot = np.array([_ols(X[idx], y[idx])[0][1:] for idx in (rng.integers(0, n, n) for _ in range(n_boot))])
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5], axis=0)

    unique_r2, vif = [], []
    for j in range(k):
        others = np.delete(X, j, axis=1)
        unique_r2.append(r2 - _ols(others, y)[2])
        vif.append(1 / (1 - _ols(others, X[:, j])[2]))

    table = pd.DataFrame({
        "factor": [FACTORS[c] for c in columns],
        "beta": coef[1:], "p_value": p_values[1:],
        "ci_low": ci_low, "ci_high": ci_high,
        "unique_r2": unique_r2, "vif": vif,
    }).sort_values("beta", key=np.abs, ascending=False).reset_index(drop=True)

    model = {
        "n": n,
        "r2": r2,
        "adj_r2": 1 - (1 - r2) * (n - 1) / dof,
        "shared_r2": r2 - sum(unique_r2),   # explained, but not attributable to any single factor
    }
    return table, model


def add_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with the score the six factors predict and the residual (actual − predicted)."""
    coef, resid, _ = _ols(df[list(FACTORS)].to_numpy(), df[SCORE].to_numpy())
    out = df.copy()
    out["predicted"] = df[SCORE] - resid
    out["residual"] = resid
    return out


def regional_residuals(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Per region: mean residual (positive = happier than its factors predict), share of
    countries above prediction, and a one-sample t-test p-value. Also returns the
    one-way ANOVA p-value for "residuals differ by region". p_adjusted corrects for
    testing every region (Bonferroni).
    """
    pred = add_predictions(df)
    groups = dict(tuple(pred.groupby(REGION)["residual"]))
    rows = [{
        REGION: region,
        "countries": len(r),
        "mean_residual": r.mean(),
        "share_above_prediction": (r > 0).mean(),
        "p_value": stats.ttest_1samp(r, 0).pvalue,
    } for region, r in groups.items()]
    anova_p = stats.f_oneway(*groups.values()).pvalue
    table = pd.DataFrame(rows).sort_values("mean_residual", ascending=False).reset_index(drop=True)
    # Bonferroni: ten regions are tested at once, so each p-value is multiplied by 10
    table["p_adjusted"] = (table["p_value"] * len(table)).clip(upper=1)
    return table, anova_p


def subgroup_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Each factor's correlation with score within a group, with p-value and 95% CI (Fisher z)."""
    n = len(df)
    rows = []
    for column, label in FACTORS.items():
        r, p = stats.pearsonr(df[column], df[SCORE])
        z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
        rows.append({"factor": label, "correlation": r, "p_value": p,
                     "ci_low": np.tanh(z - 1.96 * se), "ci_high": np.tanh(z + 1.96 * se)})
    return pd.DataFrame(rows)


def coefficient_chart(table: pd.DataFrame, model: dict) -> plt.Figure:
    """Standardized regression weights with 95% bootstrap intervals."""
    t = table.sort_values("beta")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2a9d8f" if (lo > 0 or hi < 0) else "#adb5bd" for lo, hi in zip(t["ci_low"], t["ci_high"])]
    ax.barh(t["factor"], t["beta"], color=colors)
    ax.errorbar(t["beta"], t["factor"], xerr=[t["beta"] - t["ci_low"], t["ci_high"] - t["beta"]],
                fmt="none", ecolor="black", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title=f"Effect of each factor with the other five held constant (R² = {model['r2']:.2f})",
           xlabel="Standardized coefficient (score SDs per factor SD), 95% bootstrap interval")
    return fig


def residual_chart(df: pd.DataFrame) -> plt.Figure:
    """Each country's actual minus predicted score, grouped by region."""
    pred = add_predictions(df)
    order = pred.groupby(REGION)["residual"].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=pred, x="residual", y=REGION, order=order, color="#e9ecef", fliersize=0, ax=ax)
    sns.stripplot(data=pred, x="residual", y=REGION, order=order, color="#264653", size=4, ax=ax)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set(title="Happier or less happy than the six factors predict?",
           xlabel="Actual − predicted ladder score", ylabel="")
    return fig
