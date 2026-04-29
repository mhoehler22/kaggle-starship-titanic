# Spaceship Titanic — Kaggle Competition

**Kaggle Leaderboard Score: 0.80523 | Rank: 517**

Binary classification competition hosted on Kaggle. The premise: a spaceship traveling to distant star systems collides with a space-time anomaly, transporting roughly half its 8,693 passengers to an alternate dimension. The task is to use the ship's damaged computer records to predict which passengers were transported.

[Competition page](https://www.kaggle.com/competitions/spaceship-titanic)

---

## Approach

### Missing Value Imputation
The dataset had substantial missingness across both training and test sets. Rather than dropping rows or blindly imputing column means, I used domain logic where possible:

- Passengers in CryoSleep were assumed to have zero spending.
- Passenger IDs encode a group number; passengers in the same group typically share a cabin, so group-mates were used to fill missing cabin assignments.
- Remaining continuous features (Age, spending categories) were imputed with KNN.

### Feature Engineering
- **Cabin split:** extracted Deck (letter), Cabin Number, and Side (Port/Starboard) from the raw cabin string.
- **Group features:** GroupID, GroupSize, and group-level spending aggregates.
- **Total spending:** sum across all five onboard amenities (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck).
- Categorical variables were one-hot encoded via scikit-learn's `ColumnTransformer`.

### Model Comparison
Five classifiers were trained and evaluated on a 5-fold `StratifiedKFold` cross-validation scheme:

| Model | CV Accuracy |
|---|---|
| Logistic Regression | 78.24% |
| Random Forest (600 trees) | 80.21% |
| LightGBM | 80.08% |
| XGBoost | 80.64% |
| **CatBoost** | **81.43%** |

### Hyperparameter Tuning
CatBoost was selected as the best base model and tuned with **Optuna** (Bayesian optimization, 60 trials). Best parameters found:

| Parameter | Value |
|---|---|
| iterations | 1,352 |
| learning_rate | 0.01366 |
| depth | 5 |
| l2_leaf_reg | 0.2125 |
| bagging_temperature | 2.516 |
| random_strength | 3.540 |
| border_count | 130 |

### Feature Importance (SHAP)
Spending habits were the strongest predictors — certain onboard services made a passenger more likely to have been transported, others less so. Cabin deck and home planet also carried meaningful signal.

---

## Results

The Optuna-tuned CatBoost model achieved **0.80523** on the public leaderboard, placing **517th** on the 2-month rolling leaderboard.

---

## How to Run

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place `train.csv`, `test.csv`, and `sample_submission.csv` in the project root (or use the included copies).
3. Run `spaceship_titanic_analysis.ipynb` top to bottom. The final cell writes `submission.csv`.

---

## Dependencies

- Python 3.x
- pandas, numpy, scikit-learn
- catboost, xgboost, lightgbm
- optuna
- shap
- matplotlib, seaborn

---

## Data Source

Kaggle. (2022). *Spaceship Titanic* [Machine learning competition]. https://www.kaggle.com/competitions/spaceship-titanic
