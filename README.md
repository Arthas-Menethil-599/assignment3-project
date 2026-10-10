# Linear Regression Analyzer

A Python tool for performing simple and multiple linear regression analysis on datasets. Designed for statistical research and educational purposes.

## Features

- **Simple & Multiple Linear Regression** — fit models with one or more features
- **Model Evaluation** — R², Adjusted R², MSE, RMSE, MAE
- **Coefficient Significance** — t-statistics and p-values for each predictor
- **Prediction Intervals** — compute intervals at any confidence level
- **Visualizations** — scatter plots, residual analysis, prediction interval charts, actual vs predicted plots
- **CSV Support** — load any CSV dataset via CLI or programmatically

## Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Language | Python 3.9+ | Industry standard for scientific computing; rich ecosystem |
| Numerical Computing | NumPy | Fast array operations, foundation of scientific Python |
| Data Handling | Pandas | Efficient CSV loading and data manipulation |
| Machine Learning | scikit-learn | Reliable, well-tested regression implementation |
| Statistics | SciPy | Hypothesis testing, statistical distributions |
| Visualization | Matplotlib | Publication-quality plots, highly customizable |
| Testing | pytest + pytest-cov | Simple syntax, powerful fixtures, coverage reporting |
| CI/CD | GitHub Actions | Native GitHub integration, free for public repos |

## Installation

```bash
git clone https://github.com/<your-username>/linear-regression-analyzer.git
cd linear-regression-analyzer
pip install -r requirements.txt
```

## Usage

### Quick Start (with sample data)

```bash
python main.py
```

This generates a sample housing dataset and runs the full analysis.

### Custom Dataset

```bash
python main.py --csv your_data.csv --target price --features area rooms age
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--csv` | Path to input CSV file |
| `--target` | Name of the target (dependent) column |
| `--features` | Names of feature columns (space-separated) |
| `--test-size` | Fraction of data for testing (default: 0.2) |
| `--no-plots` | Skip generating visualization plots |

### Programmatic Usage

```python
from src.regression import LinearRegressionAnalyzer
from src.visualization import RegressionPlotter

analyzer = LinearRegressionAnalyzer()
X, y = analyzer.load_csv("data.csv", target_column="price", feature_columns=["area", "rooms"])
analyzer.fit(X, y)
metrics, coef_table = analyzer.summary()

plotter = RegressionPlotter(analyzer)
plotter.scatter_with_regression_line()
plotter.residual_plot()
```

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## CI/CD Pipeline

This project uses **GitHub Actions** for continuous integration. The pipeline:

1. **Test** — runs `pytest` with coverage across Python 3.9, 3.10, 3.11, and 3.12
2. **Lint** — checks code style with `flake8`
3. **Build** — runs the demo script to verify end-to-end execution

The workflow triggers on every push and pull request to the `main` branch.

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the full configuration.

## Project Structure

```
linear-regression-analyzer/
├── .github/workflows/ci.yml   # CI/CD pipeline configuration
├── src/
│   ├── __init__.py             # Package initialization
│   ├── regression.py           # Core regression analysis module
│   └── visualization.py       # Plotting and visualization
├── tests/
│   ├── __init__.py
│   └── test_regression.py      # Unit tests
├── examples/                   # Sample data and output plots
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
