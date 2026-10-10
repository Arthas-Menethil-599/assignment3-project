"""
Main script demonstrating the Linear Regression Analyzer.

Usage:
    python main.py                          # Run with sample generated data
    python main.py --csv data.csv --target y --features x1 x2
"""

import argparse
import os
import numpy as np
import pandas as pd
from src.regression import LinearRegressionAnalyzer
from src.visualization import RegressionPlotter


def generate_sample_data(n_samples=200, noise=5.0, random_state=42):
    """Generate a sample dataset: house price prediction based on area and rooms."""
    rng = np.random.RandomState(random_state)

    area = rng.uniform(30, 200, n_samples)
    rooms = rng.randint(1, 6, n_samples).astype(float)
    price = 50 * area + 10000 * rooms + 5000 + rng.normal(0, noise * 100, n_samples)

    df = pd.DataFrame({"area_m2": area, "rooms": rooms, "price_usd": price})
    csv_path = "examples/sample_housing.csv"
    os.makedirs("examples", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Sample dataset saved to {csv_path} ({n_samples} rows)")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Linear Regression Analyzer")
    parser.add_argument("--csv", type=str, help="Path to input CSV file")
    parser.add_argument("--target", type=str, help="Name of target column")
    parser.add_argument("--features", nargs="+", help="Names of feature columns")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    analyzer = LinearRegressionAnalyzer()

    if args.csv:
        csv_path = args.csv
        target = args.target
        features = args.features
        if not target:
            parser.error("--target is required when using --csv")
    else:
        print("No CSV provided. Generating sample housing dataset...\n")
        csv_path = generate_sample_data()
        target = "price_usd"
        features = ["area_m2", "rooms"]

    X, y = analyzer.load_csv(csv_path, target_column=target, feature_columns=features)
    print(f"Loaded {len(y)} samples with {X.shape[1]} feature(s)\n")

    analyzer.fit(X, y, test_size=args.test_size)
    metrics, coef_table = analyzer.summary()

    pi = analyzer.prediction_interval(analyzer.X_test[:3])
    print("\nSample Prediction Intervals (first 3 test observations):")
    for i in range(3):
        print(f"  Observation {i+1}: {pi['prediction'][i]:.2f} "
              f"[{pi['lower_bound'][i]:.2f}, {pi['upper_bound'][i]:.2f}]")

    if not args.no_plots:
        print("\nGenerating plots...")
        plotter = RegressionPlotter(analyzer)
        plotter.scatter_with_regression_line(feature_index=0, save_path="examples/regression_line.png")
        plotter.residual_plot(save_path="examples/residuals.png")
        plotter.prediction_interval_plot(feature_index=0, save_path="examples/prediction_intervals.png")
        plotter.actual_vs_predicted(save_path="examples/actual_vs_predicted.png")
        print("Plots saved to examples/ directory.")


if __name__ == "__main__":
    main()
