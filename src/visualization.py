"""
Visualization module for Linear Regression Analyzer.

Provides plotting functions for regression results: scatter plots with
regression lines, residual plots, and prediction interval charts.
"""

import numpy as np
import matplotlib.pyplot as plt


class RegressionPlotter:
    """Generates plots for regression analysis results."""

    def __init__(self, analyzer):
        """Initialize with a fitted LinearRegressionAnalyzer instance."""
        self.analyzer = analyzer
        self.analyzer._check_fitted()

    def scatter_with_regression_line(self, feature_index=0, save_path=None):
        """Plot scatter of one feature vs target with the regression line.

        Args:
            feature_index: Index of the feature to plot (for multiple regression).
            save_path: If provided, save the figure to this path.
        """
        X_col = self.analyzer.X_test[:, feature_index]
        y_true = self.analyzer.y_test
        y_pred = self.analyzer.model.predict(self.analyzer.X_test)

        feature_name = (
            self.analyzer.feature_names[feature_index]
            if self.analyzer.feature_names
            else f"Feature {feature_index}"
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X_col, y_true, alpha=0.6, label="Actual data", color="#2196F3")

        sorted_idx = np.argsort(X_col)
        ax.plot(
            X_col[sorted_idx], y_pred[sorted_idx],
            color="#F44336", linewidth=2, label="Regression line"
        )

        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel(self.analyzer.target_name or "Target", fontsize=12)
        ax.set_title(f"Linear Regression: {feature_name} vs {self.analyzer.target_name or 'Target'}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.show()

    def residual_plot(self, save_path=None):
        """Plot residuals vs predicted values.

        Args:
            save_path: If provided, save the figure to this path.
        """
        y_pred, residuals = self.analyzer.get_residuals()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, color="#4CAF50")
        axes[0].axhline(y=0, color="#F44336", linestyle="--", linewidth=1.5)
        axes[0].set_xlabel("Predicted Values", fontsize=12)
        axes[0].set_ylabel("Residuals", fontsize=12)
        axes[0].set_title("Residuals vs Predicted Values", fontsize=14)
        axes[0].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[1].hist(residuals, bins=20, alpha=0.7, color="#9C27B0", edgecolor="white")
        axes[1].set_xlabel("Residual Value", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].set_title("Distribution of Residuals", fontsize=14)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.show()

    def prediction_interval_plot(self, feature_index=0, confidence=0.95, save_path=None):
        """Plot predictions with confidence intervals for one feature.

        Args:
            feature_index: Index of the feature to plot.
            confidence: Confidence level for the interval.
            save_path: If provided, save the figure to this path.
        """
        X_test = self.analyzer.X_test
        result = self.analyzer.prediction_interval(X_test, confidence=confidence)

        X_col = X_test[:, feature_index]
        sorted_idx = np.argsort(X_col)

        feature_name = (
            self.analyzer.feature_names[feature_index]
            if self.analyzer.feature_names
            else f"Feature {feature_index}"
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(X_col, self.analyzer.y_test, alpha=0.6, label="Actual data", color="#2196F3")
        ax.plot(
            X_col[sorted_idx], result["prediction"][sorted_idx],
            color="#F44336", linewidth=2, label="Prediction"
        )
        ax.fill_between(
            X_col[sorted_idx],
            result["lower_bound"][sorted_idx],
            result["upper_bound"][sorted_idx],
            alpha=0.2, color="#FF9800",
            label=f"{int(confidence * 100)}% Prediction Interval"
        )

        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel(self.analyzer.target_name or "Target", fontsize=12)
        ax.set_title(f"Prediction Intervals ({int(confidence * 100)}%)", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.show()

    def actual_vs_predicted(self, save_path=None):
        """Plot actual vs predicted values.

        Args:
            save_path: If provided, save the figure to this path.
        """
        y_pred = self.analyzer.model.predict(self.analyzer.X_test)
        y_true = self.analyzer.y_test

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_true, y_pred, alpha=0.6, color="#2196F3")

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color="#F44336",
                linestyle="--", linewidth=2, label="Perfect prediction")

        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.set_title("Actual vs Predicted", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.show()
