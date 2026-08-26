import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score
import shap


class LoyaltyEngine:
    def __init__(self, df: pd.DataFrame, target_col: str, id_cols: list = None):
        self.raw_df = df.copy()
        self.target_col = target_col
        self.id_cols = id_cols or []

        self.clean_df = self.raw_df.drop(columns=[c for c in self.id_cols if c in self.raw_df.columns])
        self._data_prep()

    def _data_prep(self):
        y_raw = self.clean_df[self.target_col]

        # Convert text/categorical churn labels to binary (0/1)
        if not pd.api.types.is_numeric_dtype(y_raw):
            pos_labels = {'y', 'yes', 'true', 't', '1', 'churn', 'churned', 'cancelled'}
            self.y = y_raw.astype(str).str.strip().str.lower().isin(pos_labels).astype(int)
        else:
            self.y = y_raw.astype(int)

        self.X = self.clean_df.drop(columns=[self.target_col])
        self.cat_cols = []
        self.num_cols = []

        for col in self.X.columns:
            if not pd.api.types.is_numeric_dtype(self.X[col]):
                converted = pd.to_numeric(self.X[col].astype(str).str.strip(), errors='coerce')
                if converted.notnull().sum() / len(converted) > 0.8:
                    self.X[col] = converted.fillna(converted.median())
                    self.num_cols.append(col)
                else:
                    self.X[col] = self.X[col].fillna('Missing').astype(str)
                    self.cat_cols.append(col)
            else:
                self.X[col] = self.X[col].fillna(self.X[col].median())
                self.num_cols.append(col)

    def train(self, incentive_cost: float = 15.0, customer_value: float = 70.0, acceptance_rate: float = 0.65):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=42, stratify=self.y
        )

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_weight = max(1.0, neg / max(1, pos))

        self.model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            scale_pos_weight=scale_weight,
            random_seed=42,
            verbose=False
        )

        train_pool = Pool(X_train, y_train, cat_features=self.cat_cols)
        test_pool = Pool(X_test, y_test, cat_features=self.cat_cols)
        self.model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=40)

        # Store test probability scores for dynamic threshold tuning
        self.y_test_proba = self.model.predict_proba(test_pool)[:, 1]

        # Initialize tree explainer
        self.explainer = shap.TreeExplainer(self.model)

        # Compute optimal threshold & expected profit
        self.optimize_threshold(
            incentive_cost=incentive_cost,
            customer_value=customer_value,
            acceptance_rate=acceptance_rate
        )

        return self.metrics

    def optimize_threshold(self, incentive_cost: float = 15.0, customer_value: float = 70.0, acceptance_rate: float = 0.65):
        self.incentive_cost = incentive_cost
        self.customer_value = customer_value
        self.acceptance_rate = acceptance_rate

        val_tp = (customer_value * acceptance_rate) - incentive_cost
        cost_fp = -incentive_cost
        val_tn = 0.0
        cost_fn = -customer_value

        thresholds = np.linspace(0.05, 0.95, 91)
        profits = []

        y_true = self.y_test.values if isinstance(self.y_test, pd.Series) else self.y_test

        for t in thresholds:
            preds = (self.y_test_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            profit = (tp * val_tp) + (fp * cost_fp) + (tn * val_tn) + (fn * cost_fn)
            profits.append(profit)

        best_idx = int(np.argmax(profits))
        self.optimal_threshold = float(thresholds[best_idx])
        self.max_profit = float(profits[best_idx])
        self.thresholds = thresholds
        self.profits = profits

        opt_preds = (self.y_test_proba >= self.optimal_threshold).astype(int)
        self.metrics = {
            "roc_auc": round(float(roc_auc_score(y_true, self.y_test_proba)), 4),
            "precision": round(float(precision_score(y_true, opt_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, opt_preds)), 4),
            "optimal_threshold": round(self.optimal_threshold, 2),
            "projected_savings": round(self.max_profit, 2)
        }

        return self.optimal_threshold

    def explain_single(self, input_row: pd.DataFrame):
        shap_values = self.explainer(input_row)
        return shap_values[0]