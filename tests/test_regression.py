"""Tests for the LinearRegressionAnalyzer module."""

import os
import numpy as np
import pandas as pd
import pytest
from src.regression import LinearRegressionAnalyzer


@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file with known linear relationship."""
    rng = np.random.RandomState(0)
    n = 100
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 5, n)
    y = 3 * x1 + 7 * x2 + 10 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def fitted_analyzer(sample_csv):
    """Return a fitted analyzer on sample data."""
    analyzer = LinearRegressionAnalyzer()
    X, y = analyzer.load_csv(sample_csv, target_column="y")
    analyzer.fit(X, y, test_size=0.2, random_state=42)
    return analyzer


class TestLoadCSV:
    def test_load_valid_csv(self, sample_csv):
        analyzer = LinearRegressionAnalyzer()
        X, y = analyzer.load_csv(sample_csv, target_column="y")
        assert X.shape == (100, 2)
        assert y.shape == (100,)

    def test_load_with_feature_selection(self, sample_csv):
        analyzer = LinearRegressionAnalyzer()
        X, y = analyzer.load_csv(sample_csv, target_column="y", feature_columns=["x1"])
        assert X.shape == (100, 1)

    def test_invalid_target_column(self, sample_csv):
        analyzer = LinearRegressionAnalyzer()
        with pytest.raises(ValueError, match="Target column"):
            analyzer.load_csv(sample_csv, target_column="nonexistent")

    def test_invalid_feature_column(self, sample_csv):
        analyzer = LinearRegressionAnalyzer()
        with pytest.raises(ValueError, match="Feature columns not found"):
            analyzer.load_csv(sample_csv, target_column="y", feature_columns=["bad_col"])


class TestFit:
    def test_fit_returns_coefficients(self, fitted_analyzer):
        assert fitted_analyzer.model is not None
        assert fitted_analyzer._is_fitted is True
        assert len(fitted_analyzer.model.coef_) == 2

    def test_coefficients_close_to_true_values(self, fitted_analyzer):
        coefs = fitted_analyzer.model.coef_
        intercept = fitted_analyzer.model.intercept_
        # True values: y = 3*x1 + 7*x2 + 10
        assert abs(coefs[0] - 3) < 1.0
        assert abs(coefs[1] - 7) < 1.0
        assert abs(intercept - 10) < 2.0

    def test_train_test_split(self, fitted_analyzer):
        total = len(fitted_analyzer.y_train) + len(fitted_analyzer.y_test)
        assert total == 100
        assert len(fitted_analyzer.y_test) == 20


class TestEvaluate:
    def test_evaluate_returns_metrics(self, fitted_analyzer):
        metrics = fitted_analyzer.evaluate()
        assert "r_squared" in metrics
        assert "adjusted_r_squared" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics

    def test_r_squared_high_for_linear_data(self, fitted_analyzer):
        metrics = fitted_analyzer.evaluate()
        assert metrics["r_squared"] > 0.95

    def test_rmse_is_sqrt_of_mse(self, fitted_analyzer):
        metrics = fitted_analyzer.evaluate()
        assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 1e-10


class TestResiduals:
    def test_residuals_shape(self, fitted_analyzer):
        y_pred, residuals = fitted_analyzer.get_residuals()
        assert y_pred.shape == (20,)
        assert residuals.shape == (20,)

    def test_residuals_mean_near_zero(self, fitted_analyzer):
        _, residuals = fitted_analyzer.get_residuals()
        assert abs(np.mean(residuals)) < 2.0


class TestPredict:
    def test_predict_single_observation(self, fitted_analyzer):
        result = fitted_analyzer.predict([[5.0, 2.5]])
        assert result.shape == (1,)

    def test_predict_multiple_observations(self, fitted_analyzer):
        result = fitted_analyzer.predict([[1, 1], [2, 2], [3, 3]])
        assert result.shape == (3,)

    def test_predict_before_fit_raises_error(self):
        analyzer = LinearRegressionAnalyzer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            analyzer.predict([[1, 2]])


class TestPredictionInterval:
    def test_interval_structure(self, fitted_analyzer):
        result = fitted_analyzer.prediction_interval([[5.0, 2.5]])
        assert "prediction" in result
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert "confidence" in result

    def test_lower_bound_less_than_upper(self, fitted_analyzer):
        result = fitted_analyzer.prediction_interval([[5.0, 2.5]])
        assert result["lower_bound"][0] < result["upper_bound"][0]

    def test_prediction_within_bounds(self, fitted_analyzer):
        result = fitted_analyzer.prediction_interval([[5.0, 2.5]])
        assert result["lower_bound"][0] <= result["prediction"][0] <= result["upper_bound"][0]


class TestCoefficientSignificance:
    def test_significance_table_shape(self, fitted_analyzer):
        table = fitted_analyzer.coefficient_significance()
        assert len(table) == 3  # intercept + 2 features
        assert "p_value" in table.columns

    def test_significant_coefficients(self, fitted_analyzer):
        table = fitted_analyzer.coefficient_significance()
        # All coefficients should be significant for clean linear data
        assert all(table["p_value"] < 0.05)
