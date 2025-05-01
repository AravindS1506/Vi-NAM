# Vi-NAM: Variational Inference Neural Additive Model

## Description
Vi-NAM is a Neural Additive Model (NAM) that uses variational inference (VI) to perform Bayesian neural network regression on tabular datasets. It processes feature groups independently through Bayesian neural networks, combining their contributions additively to predict a target variable. The model is trained on datasets like Yacht Hydrodynamics, Concrete Compressive Strength, Kin8nm, and Bike Sharing Demand, outputting the negative log-likelihood (NLL) for each dataset as a performance metric.

## Prerequisites
- Python 3.8+
- PyTorch
- Pyro
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Vi-NAM
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure the dataset is accessible via `sklearn.datasets.fetch_openml` or provide your own dataset in a compatible format.
2. Run the training script:
   ```bash
   python vi_nam_train.py
   ```
3. The script will:
   - Load and preprocess the datasets (Yacht Hydrodynamics, Concrete Compressive Strength, Kin8nm, Bike Sharing Demand).
   - Train the Vi-NAM model on each dataset.
   - Output training metrics (ELBO, Log Likelihood, KL Divergence, Validation RMSE, Validation NLL) every 2 epochs.
   - Compute and display the final test negative log-likelihood (NLL) for each dataset.

## Output
- **Console Output**: During training, the script prints metrics every 2 epochs, including:
  - Train Loss (ELBO)
  - Log Likelihood
  - KL Divergence
  - Validation RMSE
  - Validation NLL
  - KL Anneal Factor
  - Learning Rate
- **Final Output**: A dictionary (`avg_nll`) containing the final test NLL for each dataset, printed as:
  ```python
  {'yacht_hydrodynamics': <nll_value>, 'concrete_compressive_strength': <nll_value>, 'kin8nm': <nll_value>, 'Bike_Sharing_Demand': <nll_value>}
  ```

## Notes
- The model uses a dynamic input dimension based on the dataset and applies ordinal encoding for categorical features and standard scaling for numerical features.
- Training is configured with 500 Monte Carlo samples, 1 epoch (for demonstration; increase for better convergence), and KL annealing over 20 epochs.
- The codebase is designed for extensibility, allowing custom datasets and feature group configurations.