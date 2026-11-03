import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import happiness as hp  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return hp.load_data()


@pytest.fixture(scope="module")
def regression(df):
    table, model = hp.regression_summary(df)
    return table.set_index("factor"), model


def test_model_fit(regression):
    _, model = regression
    assert model["n"] == 149
    assert model["r2"] == pytest.approx(0.756, abs=0.001)
    assert model["adj_r2"] == pytest.approx(0.746, abs=0.001)
    assert model["shared_r2"] == pytest.approx(0.667, abs=0.001)


def test_coefficients_match_independent_lstsq(df, regression):
    table, _ = regression
    X = df[list(hp.FACTORS)]
    X = ((X - X.mean()) / X.std()).to_numpy()
    y = ((df[hp.SCORE] - df[hp.SCORE].mean()) / df[hp.SCORE].std()).to_numpy()
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0][1:]
    for label, value in zip(hp.FACTORS.values(), coef):
        assert table.loc[label, "beta"] == pytest.approx(value)


@pytest.mark.parametrize("factor, beta", [
    ("GDP", 0.302), ("Support", 0.265), ("Freedom", 0.212),
    ("Life Expectancy", 0.191), ("Perception of Corruption", -0.101), ("Generosity", 0.051),
])
def test_standardized_coefficients(regression, factor, beta):
    assert regression[0].loc[factor, "beta"] == pytest.approx(beta, abs=0.001)


def test_significance_matches_writeup(regression):
    table, _ = regression
    clearly_positive = table.loc[["GDP", "Support", "Freedom"]]
    assert (clearly_positive["p_value"] < 0.01).all() and (clearly_positive["ci_low"] > 0).all()
    assert table.loc["Generosity", "p_value"] > 0.05
    corruption = table.loc["Perception of Corruption"]
    assert corruption["p_value"] < 0.05 and corruption["ci_low"] < 0 < corruption["ci_high"]   # borderline


def test_bootstrap_is_reproducible(df):
    a, _ = hp.regression_summary(df, n_boot=200, seed=1)
    b, _ = hp.regression_summary(df, n_boot=200, seed=1)
    assert a.equals(b)


def test_freedom_has_largest_unique_share_and_vifs_ok(regression):
    table, _ = regression
    assert table["unique_r2"].idxmax() == "Freedom"
    assert (table["vif"] < 10).all()
    assert table.loc["GDP", "vif"] == pytest.approx(5.1, abs=0.05)


def test_predictions_and_residuals(df):
    pred = hp.add_predictions(df)
    assert pred["residual"].mean() == pytest.approx(0, abs=1e-9)
    assert np.allclose(pred["predicted"] + pred["residual"], pred[hp.SCORE])
    assert pred.nlargest(1, "residual")["Country name"].item() == "Congo (Brazzaville)"
    assert pred.nsmallest(1, "residual")["Country name"].item() == "Botswana"


def test_regional_residuals(df):
    table, anova_p = hp.regional_residuals(df)
    table = table.set_index(hp.REGION)
    assert anova_p < 0.001
    assert table["countries"].sum() == 149
    survivors = set(table.index[table["p_adjusted"] < 0.05])
    assert survivors == {"Latin America and Caribbean", "Western Europe"}
    se_asia = table.loc["Southeast Asia"]
    assert se_asia["mean_residual"] == pytest.approx(-0.44, abs=0.01)
    assert round(se_asia["share_above_prediction"] * 9) == 1          # 8 of 9 below prediction
    assert se_asia["p_value"] < 0.05 < se_asia["p_adjusted"]


def test_subgroup_intervals(df):
    top, bottom = hp.top_and_bottom(df, 20)
    t = hp.subgroup_correlations(top).set_index("factor")
    assert t.loc["Freedom", "ci_low"] == pytest.approx(0.43, abs=0.01)
    assert t.loc["Freedom", "ci_high"] == pytest.approx(0.89, abs=0.01)
    assert (t["ci_low"] <= t["correlation"]).all() and (t["correlation"] <= t["ci_high"]).all()
    assert (hp.subgroup_correlations(bottom)["p_value"] > 0.05).all()   # nothing significant


def test_new_charts_render(df, regression):
    table, model = hp.regression_summary(df, n_boot=50)
    hp.coefficient_chart(table, model)
    hp.residual_chart(df)
