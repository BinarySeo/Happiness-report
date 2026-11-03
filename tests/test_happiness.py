import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import happiness as hp  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return hp.load_data()


def test_loads_all_countries_with_expected_columns(df):
    assert len(df) == 149
    assert list(df.columns) == ["Country name", hp.REGION, hp.SCORE, *hp.FACTORS]
    assert df[[hp.SCORE, *hp.FACTORS]].notna().all().all()


def test_sorted_happiest_first(df):
    assert df.iloc[0]["Country name"] == "Finland"
    assert df[hp.SCORE].is_monotonic_decreasing


def test_region_scores_use_each_regions_own_countries(df):
    scores = hp.region_scores(df).set_index(hp.REGION)["avg_score"].round(2)
    # Regression test: the original chart plotted South Asia's average (4.44) as Southeast Asia
    assert scores["Southeast Asia"] == 5.41
    assert scores["South Asia"] == 4.44
    assert scores["North America and ANZ"] == 7.13
    assert len(scores) == 10


@pytest.mark.parametrize("factor, r, significant", [
    ("GDP", 0.790, True),
    ("Life Expectancy", 0.768, True),
    ("Support", 0.757, True),
    ("Freedom", 0.608, True),
    ("Perception of Corruption", -0.421, True),
    ("Generosity", -0.018, False),
])
def test_factor_correlations_match_readme(df, factor, r, significant):
    row = hp.factor_correlations(df).set_index("factor").loc[factor]
    assert row["correlation"] == pytest.approx(r, abs=0.001)
    assert bool(row["p_value"] < 0.05) == significant


def test_top_20_freedom_and_corruption(df):
    top, bottom = hp.top_and_bottom(df, 20)
    corr = hp.numeric(top).corr()[hp.SCORE]
    assert round(corr["Freedom to make life choices"], 2) == 0.73
    assert round(corr["Perceptions of corruption"], 2) == -0.65
    assert set(top["Country name"]).isdisjoint(bottom["Country name"])


def test_every_region_in_exactly_one_group(df):
    grouped = [region for regions in hp.REGION_GROUPS.values() for region in regions]
    assert sorted(grouped) == sorted(df[hp.REGION].unique())
    assert sum(len(hp.region_group(df, g)) for g in hp.REGION_GROUPS) == len(df)


@pytest.mark.parametrize("column", list(hp.FACTORS))
def test_scatter_trend_line_is_fitted_to_its_own_factor(df, column):
    # Regression test: Freedom and Corruption charts used other factors' trend lines
    fig = hp.factor_scatter(df, column)
    line = next(l for l in fig.axes[0].lines if l.get_label().startswith("trend"))
    xs, ys = line.get_data()
    slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    r = hp.factor_correlations(df).set_index("column").loc[column, "correlation"]
    assert (slope > 0) == (r > 0)
    assert f"r = {r:.2f}" in line.get_label()


def test_charts_render(df):
    hp.region_bar_chart(df)
    hp.corr_heatmap(df, "All countries")


def test_load_data_rejects_wrong_file(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing columns"):
        hp.load_data(bad)
