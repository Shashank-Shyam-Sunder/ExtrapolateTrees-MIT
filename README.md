# Extrapolation-MIT Repository

Welcome to the Extrapolation-MIT repository! This repository contains implementations and research related to machine learning methods for extrapolation, particularly focusing on Stochastic Threshold Model Trees (STMT).

## 📁 Repository Structure

This repository is organized as follows:

```
Extrapolation-MIT/
├── README.md (this file)
└── Stochastic-Threshold-Model-Trees-master_updated/    # Main project directory
    ├── README.md                                       # Detailed project documentation
    ├── LICENSE.txt                                     # MIT License
    ├── setup.py                                        # Package installation script
    ├── .gitignore                                      # Git ignore rules
    ├── stmt/                                          # Core Python package
    │   ├── __init__.py                                # Package initialization
    │   ├── criterion.py                               # Splitting criteria implementations
    │   ├── threshold_selector.py                      # Threshold selection strategies
    │   └── regressor/                                 # Regression components
    │       ├── stmt.py                                # Main STMT algorithm
    │       ├── regression_tree.py                     # Tree implementation
    │       ├── node.py                                # Tree node implementation
    │       ├── pls.py                                 # PLS regression support
    │       └── mean_regressor.py                      # Mean regression baseline
    ├── notebook/                                      # Jupyter notebook examples
    │   ├── 1dim_function.ipynb                       # 1D function approximation
    │   ├── 2dim_function.ipynb                       # 2D function approximation
    │   ├── logS.ipynb                                 # Solubility prediction example
    │   └── 1dim_comparison.png                       # Visualization example
    └── data/                                          # Example datasets
        └── logSdataset1290.csv                       # Solubility dataset
```

## 🚀 Quick Navigation Guide

### For Users Who Want to:

**🔬 Learn About the Method**
- Start with: [`Stochastic-Threshold-Model-Trees-master_updated/README.md`](Stochastic-Threshold-Model-Trees-master_updated/README.md)
- This contains comprehensive documentation about STMT, including theory, examples, and API reference

**💻 Install and Use the Package**
1. Navigate to: `Stochastic-Threshold-Model-Trees-master_updated/`
2. Follow installation instructions in the project README
3. Use the Quick Start guide for basic usage examples

**📊 See Examples and Tutorials**
- Go to: [`Stochastic-Threshold-Model-Trees-master_updated/notebook/`](Stochastic-Threshold-Model-Trees-master_updated/notebook/)
- Contains Jupyter notebooks with:
  - `1dim_function.ipynb`: Basic 1D regression example
  - `2dim_function.ipynb`: 2D function approximation
  - `logS.ipynb`: Real-world solubility prediction case study

**🔍 Explore the Source Code**
- Main algorithm: [`Stochastic-Threshold-Model-Trees-master_updated/stmt/regressor/stmt.py`](Stochastic-Threshold-Model-Trees-master_updated/stmt/regressor/stmt.py)
- Tree implementation: [`Stochastic-Threshold-Model-Trees-master_updated/stmt/regressor/regression_tree.py`](Stochastic-Threshold-Model-Trees-master_updated/stmt/regressor/regression_tree.py)
- Threshold selection: [`Stochastic-Threshold-Model-Trees-master_updated/stmt/threshold_selector.py`](Stochastic-Threshold-Model-Trees-master_updated/stmt/threshold_selector.py)

**📈 Access Example Data**
- Dataset location: [`Stochastic-Threshold-Model-Trees-master_updated/data/logSdataset1290.csv`](Stochastic-Threshold-Model-Trees-master_updated/data/logSdataset1290.csv)
- This contains a solubility prediction dataset with 1290 samples

## 🎯 What is STMT?

**Stochastic Threshold Model Trees (STMT)** is an advanced machine learning algorithm designed for regression tasks with a focus on **extrapolation** - making predictions beyond the range of training data. Unlike traditional regression trees that often fail in extrapolation scenarios, STMT maintains trend consistency and provides reliable predictions outside the training domain.

### Key Features:
- ✅ **Extrapolation-aware**: Specifically designed for out-of-domain predictions
- ✅ **Ensemble approach**: Uses multiple regression trees for robustness
- ✅ **Scikit-learn compatible**: Easy integration with existing ML pipelines
- ✅ **Flexible**: Supports any scikit-learn regressor at leaf nodes

## 🛠️ Getting Started

### Option 1: Quick Installation
```bash
# Clone the repository
git clone https://github.com/your-username/Extrapolation-MIT.git
cd Extrapolation-MIT/Stochastic-Threshold-Model-Trees-master_updated

# Install the package
pip install -e .
```

### Option 2: Direct from GitHub
```bash
pip install git+https://github.com/your-username/Extrapolation-MIT.git#subdirectory=Stochastic-Threshold-Model-Trees-master_updated
```

### Basic Usage Example
```python
from stmt.regressor.stmt import StochasticThresholdModelTrees
from stmt.threshold_selector import NormalGaussianDistribution
from stmt.criterion import MSE
from sklearn.linear_model import LinearRegression
import numpy as np

# Create and train model
model = StochasticThresholdModelTrees(
    n_estimators=100,
    criterion=MSE(),
    regressor=LinearRegression(),
    threshold_selector=NormalGaussianDistribution(5),
    random_state=42
)

# Fit and predict
X = np.random.rand(100, 3)
y = X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.1, 100)
model.fit(X, y)
predictions = model.predict(X)
```

## 📚 Documentation Levels

1. **High-Level Overview**: This README (repository navigation)
2. **Detailed Documentation**: [`Stochastic-Threshold-Model-Trees-master_updated/README.md`](Stochastic-Threshold-Model-Trees-master_updated/README.md)
3. **Code Documentation**: Inline comments and docstrings in source files
4. **Interactive Examples**: Jupyter notebooks in the `notebook/` directory

## 🔬 Research Context

This repository contains implementations related to the research paper:
> **"Stochastic Threshold Model Trees: A Tree-Based Ensemble Method for Dealing with Extrapolation"**
> 
> arXiv preprint arXiv:2009.09171 (2020)

The method addresses a critical limitation in traditional machine learning: poor extrapolation performance when making predictions outside the training data range.

## 📄 License

This project is licensed under the MIT License - see [`LICENSE.txt`](Stochastic-Threshold-Model-Trees-master_updated/LICENSE.txt) for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Need Help?

- **For usage questions**: Check the detailed README in the main project directory
- **For examples**: Explore the Jupyter notebooks
- **For technical issues**: Review the source code documentation
- **For research questions**: Refer to the original paper

---

**Start your journey**: Head to [`Stochastic-Threshold-Model-Trees-master_updated/README.md`](Stochastic-Threshold-Model-Trees-master_updated/README.md) for comprehensive documentation and examples!