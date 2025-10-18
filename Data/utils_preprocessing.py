import pandas as pd
import re
import optuna
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    confusion_matrix,
)
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone


def create_calibrated_ensemble(
    best_model_xgb: BaseEstimator,
    best_model_lgbm: BaseEstimator,
    preprocessor_xgb: Pipeline,
    preprocessor_lgbm: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[CalibratedClassifierCV, Dict[str, float]]:
    """
    Creates, calibrates, and fits an ensemble model using cloned XGBoost and LightGBM
    models within pipelines, and returns the fitted ensemble along with its
    performance metrics on the training data.

    Args:
        best_model_xgb (BaseEstimator): The best trained XGBoost model (e.g., from GridSearchCV).
        best_model_lgbm (BaseEstimator): The best trained LightGBM model.
        preprocessor_xgb (Pipeline): The fitted preprocessing pipeline for the XGBoost model.
        preprocessor_lgbm (Pipeline): The fitted preprocessing pipeline for the LightGBM model.
        X_train (pd.DataFrame): The training features DataFrame.
        y_train (pd.Series | np.ndarray): The training target Series or array.

    Returns:
        Tuple[CalibratedClassifierCV, Dict[str, float]]:
            A tuple containing:
            - calibrated_ensemble (CalibratedClassifierCV): The fitted calibrated ensemble classifier.
            - ensemble_metrics (Dict[str, float]): A dictionary of performance metrics
              ('accuracy', 'roc_auc', 'f1') on the training data.
    """
    xgb_cloned = clone(best_model_xgb)
    xgb_cloned.set_params(
        early_stopping_rounds=None, n_estimators=best_model_xgb.best_iteration
    )

    lgbm_cloned = clone(best_model_lgbm)
    lgbm_cloned.set_params(
        early_stopping_rounds=None, n_estimators=best_model_lgbm.best_iteration_
    )

    xgb_pipeline = Pipeline(
        [("preprocessing", preprocessor_xgb), ("classifier", xgb_cloned)]
    )

    lgbm_pipeline = Pipeline(
        [("preprocessing", preprocessor_lgbm), ("classifier", lgbm_cloned)]
    )

    voting_ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb_pipeline),
            ("lgbm", lgbm_pipeline),
        ],
        voting="soft",
        n_jobs=-1,
    )

    calibrated_ensemble = CalibratedClassifierCV(
        estimator=voting_ensemble, method="sigmoid", cv=3
    )

    calibrated_ensemble.fit(X_train, y_train)

    preds = calibrated_ensemble.predict(X_train)
    probs = calibrated_ensemble.predict_proba(X_train)[:, 1]

    ensemble_metrics = {
        "accuracy": accuracy_score(y_train, preds),
        "roc_auc": roc_auc_score(y_train, probs),
        "f1": f1_score(y_train, preds),
    }

    return calibrated_ensemble, ensemble_metrics


def tune_lightgbm(X, y, n_trials=50, timeout=3600):
    def objective(trial):

        numerical = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
            ]
        )
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore", drop="if_binary", sparse_output=False
                    ),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            [("num", num_pipe, numerical), ("cat", cat_pipe, categorical)]
        )

        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 2),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        val_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            X_train_prep = preprocessor.fit_transform(X_train_fold)
            X_val_prep = preprocessor.transform(X_val_fold)

            model = LGBMClassifier(
                n_estimators=1000, random_state=42, n_jobs=1, **params
            )

            model.fit(
                X_train_prep,
                y_train_fold,
                eval_set=[(X_val_prep, y_val_fold)],
                callbacks=[
                    early_stopping(stopping_rounds=50, verbose=False),
                    log_evaluation(0),
                ],
            )

            preds = model.predict_proba(X_val_prep)[:, 1]
            pred_labels = (preds > 0.5).astype(int)
            val_scores.append(accuracy_score(y_val_fold, pred_labels))

        return np.mean(val_scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True
    )

    best_params = study.best_params

    numerical = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", drop="if_binary", sparse_output=False
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [("num", num_pipe, numerical), ("cat", cat_pipe, categorical)]
    )
    X_processed = preprocessor.fit_transform(X)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_processed, y, test_size=0.2, stratify=y, random_state=42
    )

    final_model = LGBMClassifier(
        n_estimators=5000,
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=1,
        **best_params,
    )

    final_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=100, verbose=False),
            log_evaluation(0),
        ],
    )

    train_preds = final_model.predict(X_processed)
    train_probs = final_model.predict_proba(X_processed)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, train_preds),
        "roc_auc": roc_auc_score(y, train_probs),
        "f1": f1_score(y, train_preds),
        "best_params": best_params,
        "best_iteration": final_model.best_iteration_,
    }

    return final_model, preprocessor, metrics


def tune_xgboost(X, y, n_trials=50, timeout=3600):
    def objective(trial):

        numerical = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
            ]
        )

        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore", drop="if_binary", sparse_output=False
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            [("num", num_pipe, numerical), ("cat", cat_pipe, categorical)]
        )

        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "gamma": trial.suggest_float("gamma", 0.1, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 2),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        val_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            X_train_prep = preprocessor.fit_transform(X_train_fold)
            X_val_prep = preprocessor.transform(X_val_fold)

            model = XGBClassifier(
                n_estimators=1000,
                early_stopping_rounds=50,
                eval_metric="logloss",
                random_state=42,
                use_label_encoder=False,
                tree_method="hist",
                n_jobs=1,
                **params,
            )

            model.fit(
                X_train_prep,
                y_train_fold,
                eval_set=[(X_val_prep, y_val_fold)],
                verbose=0,
            )

            preds = model.predict_proba(X_val_prep)[:, 1]
            pred_labels = (preds > 0.5).astype(int)
            val_scores.append(accuracy_score(y_val_fold, pred_labels))

        return np.mean(val_scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True
    )

    best_params = study.best_params

    numerical = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", drop="if_binary", sparse_output=False
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [("num", num_pipe, numerical), ("cat", cat_pipe, categorical)]
    )

    X_processed = preprocessor.fit_transform(X)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    final_model = XGBClassifier(
        n_estimators=5000,
        early_stopping_rounds=100,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        tree_method="hist",
        n_jobs=1,
        **best_params,
    )

    final_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)

    train_preds = final_model.predict(X_processed)
    train_probs = final_model.predict_proba(X_processed)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, train_preds),
        "roc_auc": roc_auc_score(y, train_probs),
        "f1": f1_score(y, train_preds),
        "best_params": best_params,
        "best_iteration": final_model.best_iteration,
    }

    return final_model, preprocessor, metrics


def extract_title(name: str) -> str:
    """
    Extracts a common title (e.g., Mr, Mrs, Dr) from a person's name string.

    This function searches for a predefined set of titles within the input name string.
    If a title is found, it returns the first matched title.
    If no known title is found, it returns the string "None".

    Args:
        name (str): The full name of a person, from which to extract the title.

    Returns:
        str: The extracted title (e.g., "Mr", "Mrs", "Dr") or "None" if no title is found.
    """
    match = re.search(
        r"\b(Mr|Mrs|Ms|Miss|Dr|Rev|Capt|Col|Major|Sir|Lady|Jonkheer|Don|Countess)\b",
        str(name),
        re.IGNORECASE,
    )

    return match.group(0) if match else "None"


def calculate_vif(dataframe: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for numeric features in a DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        target_column (str): Optional. Column to exclude (e.g., target like 'stroke').

    Returns:
        pd.DataFrame: VIF scores sorted descending by VIF.
    """

    if target_column and target_column in dataframe.columns:
        X = dataframe.drop(columns=target_column)
    else:
        X = dataframe.copy()

    X = X.select_dtypes(include=[np.number])

    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [VIF(X.values, i) for i in range(X.shape[1])]
    vif_data["VIF"] = vif_data["VIF"].round(2)

    return vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)


def bootstrap_median_diff(
    data1, data2, num_iterations=1000, ci=95
) -> Tuple[float, float]:
    """
    Computes the bootstrap confidence interval for the difference in medians
    between two datasets.

    This function randomly resamples the two input datasets with replacement
    to simulate the sampling distribution of the difference in medians. It
    calculates the confidence interval for the difference based on the
    bootstrap resamples.

    Parameters:
    -----------
    data1 : array-like
        First dataset.
    data2 : array-like
        Second dataset.
    num_iterations : int, optional (default=1000)
        Number of bootstrap iterations to perform.
    ci : float, optional (default=95)
        The confidence level for the interval (e.g., 95 for a 95% confidence interval).

    Returns:
    --------
    lower : float
        The lower bound of the bootstrap confidence interval.
    upper : float
        The upper bound of the bootstrap confidence interval.
    """
    boot_diffs = []
    n1 = len(data1)
    n2 = len(data2)
    for i in range(num_iterations):
        boot_sample1 = np.random.choice(data1, size=n1, replace=True)
        boot_sample2 = np.random.choice(data2, size=n2, replace=True)
        boot_diffs.append(np.median(boot_sample1) - np.median(boot_sample2))
    lower = np.percentile(boot_diffs, (100 - ci) / 2)
    upper = np.percentile(boot_diffs, 100 - (100 - ci) / 2)
    return lower, upper


def crosstab_chi2_test(df: pd.DataFrame, col_x: str, col_y: str) -> None:
    """
    Computes and prints a normalized crosstab and performs Chi-squared test of independence.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data
        col_x (str): The name of the row variable (e.g., 'FrequentFlyer')
        col_y (str): The name of the column variable (e.g., 'TravelInsurance')
    """
    print(f"\n Crosstab of {col_x} vs. {col_y}\n")

    cross_tab_norm = pd.crosstab(df[col_x], df[col_y], normalize="index").round(2) * 100
    print("Normalized Crosstab (%):")
    print(cross_tab_norm)

    cross_tab_counts = pd.crosstab(df[col_x], df[col_y])
    chi2, p, dof, expected = chi2_contingency(cross_tab_counts)

    print("\nChi-squared test results:")
    print(f"Chi-squared Statistic: {chi2:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-value: {p:.4f}")

    if p < 0.05:
        print(" Statistically significant association (p < 0.05)")
    else:
        print(" No significant association (p ≥ 0.05)")
