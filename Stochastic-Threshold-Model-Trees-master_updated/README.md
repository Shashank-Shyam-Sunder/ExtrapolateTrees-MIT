# Stochastic Threshold Model Trees (STMT)

A Python implementation of Stochastic Threshold Model Trees, an ensemble tree-based regression method designed to provide reasonable extrapolation predictions for physicochemical and other data with expected monotonicity properties.

## Overview

Stochastic Threshold Model Trees (STMT) is an advanced machine learning algorithm that extends traditional tree-based ensemble methods to handle extrapolation scenarios more effectively. Unlike standard regression trees that may produce poor predictions outside the training domain, STMT incorporates stochastic threshold selection to maintain trend consistency in extrapolation regions.

### Key Features

- **Extrapolation-aware**: Designed specifically to handle predictions beyond the training data range
- **Ensemble approach**: Uses multiple regression trees for robust predictions
- **Flexible regressors**: Supports any scikit-learn compatible regressor at leaf nodes
- **Customizable thresholds**: Configurable threshold selection strategies
- **Scikit-learn compatible**: Follows scikit-learn's estimator interface

## Requirements

### Core Dependencies
- Python >= 3.6
- [NumPy](https://numpy.org/) >= 1.17
- [scikit-learn](https://scikit-learn.org/stable/) >= 0.21
- [joblib](https://pypi.org/project/joblib/) >= 0.13

### Optional Dependencies (for examples and notebooks)
- [pandas](https://pandas.pydata.org/) >= 0.25
- [matplotlib](https://matplotlib.org/) >= 3.1
- [seaborn](https://seaborn.pydata.org/) >= 0.9

## Installation

### From Source
Clone this repository and install:

```bash
git clone https://github.com/funatsu-lab/Stochastic-Threshold-Model-Trees.git
cd Stochastic-Threshold-Model-Trees
pip install -e .
```

### Direct Installation
```bash
pip install git+https://github.com/funatsu-lab/Stochastic-Threshold-Model-Trees.git
```

## Quick Start

### Basic Usage

```python
from stmt.regressor.stmt import StochasticThresholdModelTrees
from stmt.threshold_selector import NormalGaussianDistribution
from stmt.criterion import MSE
from sklearn.linear_model import LinearRegression
import numpy as np

# Create sample data
X = np.random.rand(100, 3)
y = X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.1, 100)

# Initialize the model
model = StochasticThresholdModelTrees(
    n_estimators=100,
    criterion=MSE(),
    regressor=LinearRegression(),
    threshold_selector=NormalGaussianDistribution(5),
    random_state=42
)

# Fit and predict
model.fit(X, y)
predictions = model.predict(X)
```

### Complete Example with Data Loading

```python
from stmt.regressor.stmt import StochasticThresholdModelTrees
from stmt.threshold_selector import NormalGaussianDistribution
from stmt.criterion import MSE
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('./data/logSdataset1290.csv', index_col=0)
X = data[data.columns[1:]]
y = data[data.columns[0]]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Configure model
model = StochasticThresholdModelTrees(
    n_estimators=100,              # Number of trees in the ensemble
    criterion=MSE(),               # Splitting criterion
    regressor=LinearRegression(),   # Leaf node regressor
    threshold_selector=NormalGaussianDistribution(5),  # Threshold selection strategy
    min_samples_leaf=1,            # Minimum samples per leaf
    max_features='auto',           # Features to consider for splitting
    f_select=True,                 # Enable feature selection
    ensemble_pred='mean',          # Ensemble aggregation method ('mean' or 'median')
    scaling=False,                 # Enable feature scaling at leaf nodes
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Get predictions with uncertainty
y_pred_mean, y_pred_std = model.predict(X_test, return_std=True)
```

## Parameters

### StochasticThresholdModelTrees Parameters

- **n_estimators** (int, default=100): Number of trees in the ensemble
- **criterion**: Splitting criterion object (e.g., MSE())
- **regressor**: Regression model for leaf nodes (any scikit-learn regressor)
- **threshold_selector**: Strategy for threshold selection
- **max_depth** (int, default=None): Maximum depth of trees
- **min_samples_split** (int, default=2): Minimum samples required to split
- **min_samples_leaf** (int, default=1): Minimum samples required at leaf
- **max_features** (str/int/float, default='auto'): Features to consider for splitting
- **f_select** (bool, default=True): Enable feature selection
- **ensemble_pred** (str, default='mean'): Aggregation method ('mean' or 'median')
- **scaling** (bool, default=False): Enable feature scaling at leaves
- **bootstrap** (bool, default=True): Use bootstrap sampling
- **random_state** (int, default=None): Random seed for reproducibility

## Examples and Visualizations

The method demonstrates superior extrapolation performance compared to traditional approaches:

![discontinuous_Proposed_5sigma](https://user-images.githubusercontent.com/49966285/86465964-ad039700-bd6d-11ea-80b0-8035fc726228.png)

![Sphere_Proposed_MLR_noise_scaling](https://user-images.githubusercontent.com/49966285/86466391-7d08c380-bd6e-11ea-879c-8e9b3f9ba493.png)

![1dim_comparison](https://user-images.githubusercontent.com/49966285/86992420-69c97e00-c1dc-11ea-8e2f-8b3d08ce27d4.png)

## Jupyter Notebooks

Explore the `notebook/` directory for detailed examples:
- `1dim_function.ipynb`: One-dimensional function approximation
- `2dim_function.ipynb`: Two-dimensional function approximation  
- `logS.ipynb`: Solubility prediction example

## API Methods

### Main Methods

- **`fit(X, y)`**: Train the STMT model on training data
- **`predict(X, return_std=False)`**: Make predictions on new data
  - Set `return_std=True` to get both mean predictions and standard deviation
- **`get_params(deep=True)`**: Get model parameters
- **`set_params(**params)`**: Set model parameters
- **`count_selected_feature()`**: Count features used for tree splitting

## Project Structure

```
├── stmt/                     # Main package
│   ├── regressor/           # Regression components
│   │   ├── stmt.py         # Main STMT class
│   │   ├── regression_tree.py  # Tree implementation
│   │   ├── node.py         # Tree node implementation
│   │   └── ...
│   ├── criterion.py        # Splitting criteria
│   └── threshold_selector.py  # Threshold selection strategies
├── data/                   # Example datasets
├── notebook/               # Jupyter notebook examples
└── setup.py               # Package setup
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Troubleshooting

### Common Issues

- **Import errors**: Ensure all dependencies are installed with correct versions
- **Memory issues**: Reduce `n_estimators` for large datasets
- **Slow training**: Consider reducing `max_features` or enabling `bootstrap=False`

### Performance Tips

- Use `ensemble_pred='median'` for more robust predictions
- Enable `scaling=True` for datasets with different feature scales
- Set appropriate `min_samples_leaf` to prevent overfitting

## Citation

If you use this software in your research, please cite:

```bibtex
@article{numata2020stochastic,
  title={Stochastic Threshold Model Trees: A Tree-Based Ensemble Method for Dealing with Extrapolation},
  author={Numata, Kohei and others},
  journal={arXiv preprint arXiv:2009.09171},
  year={2020}
}
```

## References

- [Original Paper](https://arxiv.org/abs/2009.09171): Stochastic Threshold Model Trees: A Tree-Based Ensemble Method for Dealing with Extrapolation
- [scikit-learn](https://scikit-learn.org/): Machine learning library in Python

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments

- Original implementation by Kohei Numata
- Built on top of scikit-learn and NumPy