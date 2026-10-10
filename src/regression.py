"""
Linear Regression Analyzer module.

Provides functionality for fitting simple and multiple linear regression models,
computing regression statistics, and making predictions with confidence intervals.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


class LinearRegressionAnalyzer:
    """Analyzer for simple and multiple linear regression."""

    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_name = None
        self._is_fitted = False

    def load_csv(self, filepath, target_column, feature_columns=None):
        """Load dataset from a CSV file.

        Args:
            filepath: Path to the CSV file.
            target_column: Name of the target (dependent) variable column.
            feature_columns: List of feature column names. If None, all columns
                except the target are used.

        Returns:
            Tuple of (X, y) as numpy arrays.
        """
        df = pd.read_csv(filepath)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]

        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")

        self.feature_names = feature_columns
        self.target_name = target_column

        X = df[feature_columns].values
        y = df[target_column].values

        return X, y

    def fit(self, X, y, test_size=0.2, random_state=42):
        """Fit the linear regression model.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
            test_size: Fraction of data reserved for testing.
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary with model coefficients and intercept.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self._is_fitted = True

        return {
            "intercept": self.model.intercept_,
            "coefficients": self.model.coef_,
            "feature_names": self.feature_names,
        }

    def evaluate(self):
        """Evaluate the fitted model on the test set.

        Returns:
            Dictionary with R2, adjusted R2, MSE, RMSE, and MAE.
        """
        self._check_fitted()

        y_pred = self.model.predict(self.X_test)
        n = len(self.y_test)
        p = self.X_test.shape[1]

        r2 = r2_score(self.y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else float("nan")
        mse = mean_squared_error(self.y_test, y_pred)

        return {
            "r_squared": r2,
            "adjusted_r_squared": adj_r2,
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(self.y_test, y_pred),
        }

    def get_residuals(self):
        """Compute residuals on the test set.

        Returns:
            Tuple of (y_predicted, residuals).
        """
        self._check_fitted()
        y_pred = self.model.predict(self.X_test)
        residuals = self.y_test - y_pred
        return y_pred, residuals

    def predict(self, X_new):
        """Make predictions for new data.

        Args:
            X_new: Feature matrix for new observations.

        Returns:
            Predicted values as numpy array.
        """
        self._check_fitted()
        X_new = np.atleast_2d(X_new)
        return self.model.predict(X_new)

    def prediction_interval(self, X_new, confidence=0.95):
        """Compute prediction intervals for new observations.

        Args:
            X_new: Feature matrix for new observations.
            confidence: Confidence level (default 0.95).

        Returns:
            Dictionary with predicted values, lower and upper bounds.
        """
        self._check_fitted()
        X_new = np.atleast_2d(X_new)

        y_pred_train = self.model.predict(self.X_train)
        residuals = self.y_train - y_pred_train
        se = np.std(residuals, ddof=self.X_train.shape[1] + 1)

        y_pred = self.model.predict(X_new)
        n = len(self.y_train)
        dof = n - self.X_train.shape[1] - 1
        t_value = stats.t.ppf((1 + confidence) / 2, dof)
        margin = t_value * se

        return {
            "prediction": y_pred,
            "lower_bound": y_pred - margin,
            "upper_bound": y_pred + margin,
            "confidence": confidence,
        }

    def coefficient_significance(self):
        """Compute t-statistics and p-values for each coefficient.

        Returns:
            DataFrame with coefficients, standard errors, t-values, and p-values.
        """
        self._check_fitted()

        y_pred = self.model.predict(self.X_train)
        residuals = self.y_train - y_pred
        n, p = self.X_train.shape
        dof = n - p - 1

        mse = np.sum(residuals ** 2) / dof
        X_with_intercept = np.column_stack([np.ones(n), self.X_train])
        cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se = np.sqrt(np.diag(cov_matrix))

        all_coefs = np.concatenate([[self.model.intercept_], self.model.coef_])
        t_values = all_coefs / se
        p_values = 2 * stats.t.sf(np.abs(t_values), dof)

        names = ["intercept"] + (self.feature_names or [f"x{i}" for i in range(p)])

        return pd.DataFrame({
            "feature": names,
            "coefficient": all_coefs,
            "std_error": se,
            "t_value": t_values,
            "p_value": p_values,
        })

    def summary(self):
        """Print a full regression summary."""
        self._check_fitted()

        metrics = self.evaluate()
        coef_table = self.coefficient_significance()

        print("=" * 60)
        print("         LINEAR REGRESSION SUMMARY")
        print("=" * 60)
        print(f"  Target variable : {self.target_name or 'y'}")
        print(f"  Training samples: {len(self.y_train)}")
        print(f"  Test samples    : {len(self.y_test)}")
        print(f"  Features        : {self.X_train.shape[1]}")
        print("-" * 60)
        print("  MODEL METRICS (Test Set)")
        print(f"  R2              : {metrics['r_squared']:.4f}")
        print(f"  Adjusted R2     : {metrics['adjusted_r_squared']:.4f}")
        print(f"  MSE             : {metrics['mse']:.4f}")
        print(f"  RMSE            : {metrics['rmse']:.4f}")
        print(f"  MAE             : {metrics['mae']:.4f}")
        print("-" * 60)
        print("  COEFFICIENTS")
        print(coef_table.to_string(index=False))
        print("=" * 60)

        return metrics, coef_table

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
