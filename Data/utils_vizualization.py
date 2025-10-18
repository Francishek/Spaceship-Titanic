import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
import statsmodels.api as sm
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
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


def plot_calibration_comparison(
    models: List[BaseEstimator],
    preprocessors: List[Optional[Union[Pipeline, ColumnTransformer]]],
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    model_names: Optional[List[str]] = None,
    n_bins: int = 10,
) -> None:
    """
    Plots calibration curves for multiple models on a single plot.

    Parameters:
    - models: list of fitted models
    - preprocessors: list of preprocessors (or None) matching models
    - X_test: test features
    - y_test: test labels
    - model_names: list of model names
    - n_bins: number of bins for calibration curve

    Returns:
        None: Displays the calibration curve comparison plot.
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]

    plt.figure(figsize=(7, 6))

    for model, preprocessor, name in zip(models, preprocessors, model_names):

        X_test_processed = preprocessor.transform(X_test) if preprocessor else X_test

        probs = model.predict_proba(X_test_processed)[:, 1]

        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=n_bins)

        plt.plot(prob_pred, prob_true, marker="o", label=name)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Probability")
    plt.title("Calibration Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_model_on_test(
    model: BaseEstimator,
    preprocessor: Optional[Union[Pipeline, ColumnTransformer]],
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Evaluates a machine learning model on the test set, with optional preprocessing.

    This function calculates and prints standard classification metrics (accuracy,
    ROC AUC, F1-score, precision, recall), generates and displays confusion matrices
    (absolute and percentage), and plots a calibration curve to assess the model's
    probability predictions.

    Args:
        model (BaseEstimator): The trained machine learning model (e.g., a scikit-learn
                                classifier or a Pipeline containing a classifier).
        preprocessor (Optional[Union[Pipeline, ColumnTransformer]]): The fitted preprocessing
                                                                      pipeline (e.g., ColumnTransformer,
                                                                      Pipeline) to transform X_test.
                                                                      If None, X_test is used directly.
        X_test (pd.DataFrame): The test features DataFrame.
        y_test (pd.Series | np.ndarray): The true labels for the test set.
        model_name (str, optional): A descriptive name for the model, used in print statements
                                    and plot titles. Defaults to "Model".

    Returns:
        Dict[str, float]: A dictionary containing the calculated performance metrics.
                          Keys include "accuracy", "roc_auc", "f1", "precision", "recall".
    """
    if preprocessor:
        X_test_processed = preprocessor.transform(X_test)
    else:
        X_test_processed = X_test

    test_preds = model.predict(X_test_processed)
    test_probs = model.predict_proba(X_test_processed)[:, 1]

    test_metrics = {
        "accuracy": accuracy_score(y_test, test_preds),
        "roc_auc": roc_auc_score(y_test, test_probs),
        "f1": f1_score(y_test, test_preds, pos_label=1),
        "precision": precision_score(y_test, test_preds, pos_label=1),
        "recall": recall_score(y_test, test_preds, pos_label=1),
    }

    print(f"\n=== Test Set Evaluation: {model_name} ===")
    for k, v in test_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    cm = confusion_matrix(y_test, test_preds)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f"Confusion Matrix ({model_name})")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", ax=axes[1])
    axes[1].set_title(f"Confusion Matrix (%) ({model_name})")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    plt.tight_layout()
    plt.show()

    return test_metrics


def plot_model_metrics_comparison(
    metrics_xgb: Dict[str, float],
    metrics_lgbm: Dict[str, float],
    ensemble_metrics: Dict[str, float],
) -> None:
    """
    Plots a comparison of key performance metrics for XGBoost, LightGBM, and an Ensemble model.

    The plot visualizes Accuracy, ROC AUC, and F1-score for class 1.

    Args:
        metrics_xgb (Dict[str, float]): Dictionary containing performance metrics for the XGBoost model.
                                        Expected keys: 'accuracy', 'roc_auc', 'f1'.
        metrics_lgbm (Dict[str, float]): Dictionary containing performance metrics for the LightGBM model.
                                         Expected keys: 'accuracy', 'roc_auc', 'f1'.
        ensemble_metrics (Dict[str, float]): Dictionary containing performance metrics for the Ensemble model.
                                            Expected keys: 'accuracy', 'roc_auc', 'f1'.

    Returns:
        None: Displays the comparison plot.
    """
    models = ["XGBoost", "LightGBM", "Ensemble"]
    metric_names = ["Accuracy", "ROC AUC", "F1 (1)"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    values = {
        "Accuracy": [
            metrics_xgb["accuracy"],
            metrics_lgbm["accuracy"],
            ensemble_metrics["accuracy"],
        ],
        "ROC AUC": [
            metrics_xgb["roc_auc"],
            metrics_lgbm["roc_auc"],
            ensemble_metrics["roc_auc"],
        ],
        "F1 (1)": [metrics_xgb["f1"], metrics_lgbm["f1"], ensemble_metrics["f1"]],
    }

    fig, axes = plt.subplots(ncols=3, figsize=(18, 5))

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        bars = values[metric]
        x = np.arange(len(models))

        ax.bar(x, bars, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=15)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} Comparison", fontsize=20)
        ax.set_ylim(0.75, 0.95)
        for j, val in enumerate(bars):
            ax.text(
                x[j], val + 0.005, f"{val:.4f}", ha="center", va="bottom", fontsize=15
            )

    plt.tight_layout()
    plt.show()


def analyze_shap(
    model: BaseEstimator,
    preprocessor: Pipeline,
    X_train: pd.DataFrame,
    model_name: str = "",
) -> Tuple[shap.Explanation, np.ndarray]:
    """
    Analyze and plot SHAP values with proper feature names

    Parameters:
    - model: The trained model (XGBoost/LightGBM)
    - preprocessor: The fitted preprocessing pipeline
    - X_train: Training data (before preprocessing)
    - model_name: Name for plot titles

    Returns:
    - Tuple[shap.Explanation, np.ndarray]:
        A tuple containing:
        - shap_values (shap.Explanation): The calculated SHAP explanation object.
        - feature_names (np.ndarray): An array of the feature names used by the model.
    """

    def get_feature_names(preprocessor):
        """Extracts feature names after preprocessing"""
        numeric_features = preprocessor.named_transformers_[
            "num"
        ].get_feature_names_out(
            input_features=X_train.select_dtypes(include=["int64", "float64"]).columns
        )

        categorical_features = preprocessor.named_transformers_[
            "cat"
        ].get_feature_names_out(
            input_features=X_train.select_dtypes(include=["object", "category"]).columns
        )

        return np.concatenate([numeric_features, categorical_features])

    full_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    X_processed = preprocessor.transform(X_train)

    feature_names = get_feature_names(preprocessor)

    explainer = shap.Explainer(
        full_pipeline.named_steps["model"],
        masker=X_processed[:100],
        feature_names=feature_names,
    )

    shap_values = explainer(X_processed[:100])

    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title(f"{model_name} - Global Feature Importance")
    plt.subplot(1, 2, 2)
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title(f"{model_name} - SHAP Value Distribution")
    plt.tight_layout()
    plt.show()

    return shap_values, feature_names


def plot_metrics_comparison(
    overall_metrics: pd.DataFrame,
    tuned_metrics_xgb: dict[str, float],
    tuned_metrics_lgbm: dict[str, float],
) -> None:
    """
    Dynamically plots comparison of metrics before and after tuning
    with metrics ordered as: Accuracy, ROC AUC, F1 (1)

    Parameters:
    - overall_metrics: DataFrame with initial metrics (pre-tuning)
    - tuned_metrics_xgb: Dict with XGBoost tuned metrics
    - tuned_metrics_lgbm: Dict with LightGBM tuned metrics
    """
    models = ["XGBoost", "LightGBM"]

    tuned_data = {
        "XGBoost": {
            "Accuracy": tuned_metrics_xgb["accuracy"],
            "ROC AUC": tuned_metrics_xgb["roc_auc"],
            "F1 (1)": tuned_metrics_xgb["f1"],
        },
        "LightGBM": {
            "Accuracy": tuned_metrics_lgbm["accuracy"],
            "ROC AUC": tuned_metrics_lgbm["roc_auc"],
            "F1 (1)": tuned_metrics_lgbm["f1"],
        },
    }
    tuned_df = pd.DataFrame.from_dict(tuned_data, orient="index")

    desired_order = ["Accuracy", "ROC AUC", "F1 (1)"]

    common_metrics = [
        m
        for m in desired_order
        if m in overall_metrics.columns and m in tuned_df.columns
    ]

    fig, axes = plt.subplots(nrows=1, ncols=len(common_metrics), figsize=(15, 5))
    if len(common_metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(common_metrics):
        before_values = overall_metrics.loc[models, metric].values
        after_values = tuned_df.loc[models, metric].values
        x = np.arange(len(models))
        width = 0.35
        axes[i].bar(x - width / 2, before_values, width, label="Before Tuning")
        axes[i].bar(x + width / 2, after_values, width, label="After Tuning")
        axes[i].set_xlabel("Model")
        axes[i].set_ylabel(metric)
        axes[i].set_title(f"{metric} Comparison")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(models)
        axes[i].legend()

        for j in range(len(models)):
            axes[i].text(
                x[j] - width / 2,
                before_values[j] + 0.01,
                f"{before_values[j]:.4f}",
                ha="center",
                va="bottom",
            )
            axes[i].text(
                x[j] + width / 2,
                after_values[j] + 0.01,
                f"{after_values[j]:.4f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    plt.show()


def plot_phik_correlation_heatmap(
    df: pd.DataFrame,
    figsize: tuple = (10, 6),
) -> None:
    """
    Generates and displays a heatmap of the Phi-k correlation matrix
    for a given DataFrame. Only the lower triangle of the matrix is displayed

    Args:
        df (pd.DataFrame): The input DataFrame for which to calculate
                           and visualize the Phi-k correlation.
        figsize (tuple, optional): A tuple (width, height) in inches
                                   for the figure size of the heatmap.
                                   Defaults to (10, 8).

    Returns:
        None: Displays the heatmap plot.
    """

    try:
        phik_corr = df.phik_matrix()
    except Exception as e:
        print(f"Error calculating Phi-k matrix: {e}")
        print(
            "Please ensure your DataFrame is suitable for phik_matrix() "
            "and the 'phik' library is correctly installed."
        )
        return

    mask = np.triu(np.ones_like(phik_corr, dtype=bool))

    plt.figure(figsize=figsize)

    sns.heatmap(
        phik_corr,
        mask=mask,
        annot=True,
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Phi-k Correlation (Lower triangle)", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_transported_distribution(df: pd.DataFrame) -> None:
    """
    Generates and displays a pie chart showing the distribution of the
    'Transported' status in a DataFrame.

    The chart includes percentages and the exact count for each category
    (True/False or 0/1).

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'Transported' column.
                          The 'Transported' column is expected to contain
                          boolean (True/False) or numerical (0/1) values.

    Returns:
        None: Displays the pie chart.
    """
    transported_counts = df["Transported"].value_counts()
    labels = transported_counts.index.astype(str)
    sizes = transported_counts.values

    def autopct_with_counts(pct: float) -> str:
        """
        Custom autopct function for matplotlib.pyplot.pie.
        Formats the label to show both percentage and the absolute count.
        """
        total = sum(sizes)
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({count})"

    plt.figure(figsize=(4, 4))
    plt.pie(
        sizes,
        labels=labels,
        autopct=autopct_with_counts,
        startangle=90,
        textprops={"fontsize": 12},
    )
    plt.title("Transported Status Distribution", fontsize=14)
    plt.axis("equal")
    plt.show()


def stacked_bar_with_percent(
    data: pd.DataFrame,
    column_x: str,
    column_y: str = "Transported",
    figsize: Tuple[int, int] = (8, 4),
) -> None:
    """
    Plot stacked bar chart with bars sized by actual frequency and
    annotated with percentages for binary target analysis.
    """
    # Get counts
    count_table = pd.crosstab(data[column_x], data[column_y])

    # Normalize by all data (not index)
    percent_table = count_table.div(count_table.sum(axis=1), axis=0) * 100

    ax = count_table.plot(
        kind="bar",
        stacked=True,
        figsize=figsize,
    )

    # Annotate with percentage labels
    for i, category in enumerate(count_table.index):
        bottom = 0
        for stroke_value in count_table.columns:
            count = count_table.loc[category, stroke_value]
            percent = percent_table.loc[category, stroke_value]
            if count > 0:
                ax.text(
                    i,
                    bottom + count / 2,
                    f"{percent:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
                )
            bottom += count

    plt.title(f"Transported by {column_x}", pad=15)
    plt.xlabel(column_x)
    plt.ylabel("Number of persons")
    plt.legend(
        title="Transported",
        labels=["No", "Yes"],
        loc="upper right",
        frameon=True,
    )
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_distribution_numerical(
    data: pd.DataFrame,
    column: str,
    target: str = "Transported",
    figsize: Tuple[int, int] = (8, 6),
    bins: int = 10,
    log_scale: bool = False,
) -> None:
    """
    Plot comparative analysis of a numerical feature vs target variable
    including general and conditional distributions and boxplots.

    Parameters:
    - data: Input DataFrame
    - column: Numerical column to analyze
    - target: Target variable name (default: Transported)
    - figsize: Figure size (width, height)
    - bins: Number of histogram bins (default: 10)
    - log_scale: Whether to apply log1p scale to the column (default: False)
    """
    col_data = np.log1p(data[column]) if log_scale else data[column]
    plot_data = data.copy()
    plot_data["_col_"] = col_data

    plt.figure(figsize=figsize)

    plt.subplot(2, 2, 1)
    sns.boxplot(
        y=plot_data["_col_"],
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )
    plt.title(f"General Boxplot of {column}" + (" (log scale)" if log_scale else ""))
    plt.xlabel("")
    plt.ylabel(column)

    # Boxplot by target
    plt.subplot(2, 2, 2)
    sns.boxplot(
        x=target,
        y="_col_",
        data=plot_data,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )
    plt.title(
        f"{column.capitalize()} by {target}" + (" (log scale)" if log_scale else "")
    )
    plt.xlabel(target)
    plt.ylabel(column)

    plt.subplot(2, 2, 3)
    sns.histplot(plot_data["_col_"].dropna(), bins=bins, kde=True, color="skyblue")
    plt.title(f"Distribution of {column}" + (" (log scale)" if log_scale else ""))
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 4)
    sns.histplot(
        data=plot_data,
        x="_col_",
        hue=target,
        bins=bins,
        kde=True,
        element="step",
        common_norm=False,
    )
    plt.title(
        f"{column.capitalize()} distribution by {target}"
        + (" (log scale)" if log_scale else "")
    )
    plt.xlabel(column)
    plt.ylabel("Density")

    plt.tight_layout()
    plt.show()
