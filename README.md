# Predicting Student Health Risk

A machine learning project that classifies a person's health condition — **at-risk**, **unhealthy**, or **fit** — from lifestyle and biometric data such as sleep, heart rate, BMI, exercise habits, diet, and stress level.

## Dataset

`playground/train.csv` and `playground/test.csv` contain the following features:

| Column | Description |
|---|---|
| `sleep_duration` | Hours of sleep |
| `heart_rate` | Resting heart rate |
| `bmi` | Body mass index |
| `calorie_expenditure` | Calories burned |
| `step_count` | Daily step count |
| `exercise_duration` | Minutes of exercise |
| `water_intake` | Daily water intake |
| `diet_type` | Diet category (e.g. veg/non-veg) |
| `stress_level` | Categorical stress level |
| `sleep_quality` | Categorical sleep quality |
| `physical_activity_level` | e.g. sedentary/active |
| `smoking_alcohol` | Smoking/alcohol use (yes/no) |
| `gender` | Gender |

Target: `health_condition` (`at-risk`, `unhealthy`, `fit`) — highly imbalanced, with `at-risk` making up ~86% of the training data.

## Approach

1. **EDA** — missing value analysis and profiling with `sweetviz` (`profile.html`).
2. **Preprocessing** — mode/median imputation for categorical/continuous features, ordinal encoding for categoricals, min-max scaling for continuous features.
3. **Modeling** — benchmarked several classifiers (Logistic Regression, SVM, Naive Bayes, Decision Tree, KNN, Random Forest, AdaBoost, Bagging, Extra Trees, Gradient Boosting, XGBoost, LightGBM, and a Keras neural net), evaluated with `StratifiedKFold` cross-validation to account for class imbalance.
4. **Scoring** — balanced accuracy. The best-performing model, LightGBM, reached **0.95 balanced accuracy**.

All exploration and modeling code lives in `main.ipynb`.

## Project structure

```
main.ipynb          # EDA, preprocessing, and modeling notebook
playground/          # train.csv, test.csv, sample_submission.csv
profile.html          # sweetviz EDA report
submission*.csv       # model prediction outputs
pyproject.toml, uv.lock, requirements.txt
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management (Python >= 3.13):

```bash
uv sync
```

Then open `main.ipynb` in Jupyter/VS Code to run the pipeline.

## License

MIT — see [LICENSE](LICENSE).