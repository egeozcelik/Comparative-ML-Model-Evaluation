import pandas as pd
import joblib
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTuning:
    """
    Model Tuning Module
    Handles hyperparameter optimization for the best performing model
    """
    
    def __init__(self, X_train, X_test, y_train, y_test, 
                 X_train_scaled, X_test_scaled, scaler,
                 model_name="CatBoost", param_grid=None, 
                 search_config=None, random_state=42):
        """
        Args:
            X_train: Training features (unscaled)
            X_test: Test features (unscaled)
            y_train: Training target
            y_test: Test target
            X_train_scaled: Scaled training features
            X_test_scaled: Scaled test features
            scaler: Fitted scaler object
            model_name: Name of the model to tune
            param_grid: Parameter grid for hyperparameter search
            search_config: RandomizedSearch configuration
            random_state: Random state for reproducibility
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.scaler = scaler
        
        self.model_name = model_name
        self.param_grid = param_grid or {}
        self.search_config = search_config or {}
        self.random_state = random_state
        
        self.best_model = None
        self.base_model = None
        self.base_cv_results = None
    
    def tune_model(self):
        """Perform hyperparameter tuning using RandomizedSearchCV"""
        self.base_model = CatBoostClassifier(verbose=False, random_state=self.random_state)
        
        print(f"\n  Baseline {self.model_name} performance:")
        cv_results = cross_validate(
            self.base_model,
            self.X_train_scaled,
            self.y_train,
            cv=5,
            scoring=['accuracy', 'f1'],
            n_jobs=-1
        )
        
        base_cv_f1 = cv_results['test_f1'].mean()
        base_cv_acc = cv_results['test_accuracy'].mean()
        self.base_cv_results = cv_results
        
        print(f"    • CV F1 Score: {base_cv_f1:.3f}")
        print(f"    • CV Accuracy: {base_cv_acc:.3f}")
        
        print(f"\n  Running RandomizedSearchCV...")
        print(f"  ◦ Search space: {len(self.param_grid)} parameters")
        print(f"  ◦ Iterations: {self.search_config['n_iter']}")
        print(f"  ◦ Cross-validation: {self.search_config['cv']}-fold")
        print(f"  ◦ Total fits: {self.search_config['n_iter'] * self.search_config['cv']}")
        
        random_search = RandomizedSearchCV(
            estimator=self.base_model,
            param_distributions=self.param_grid,
            **self.search_config
        )
        
        random_search.fit(self.X_train_scaled, self.y_train)
        
        self.best_model = random_search.best_estimator_
        
        print(f"\n  Best hyperparameters found: ✓")
        for param, value in random_search.best_params_.items():
            print(f"    • {param}: {value}")
        
        self._evaluate_on_test_set(base_cv_f1)
        self._display_classification_report()
        
        return self.best_model
    
    def _evaluate_on_test_set(self, base_cv_f1):
        """Evaluate tuned model on test set"""
        import warnings
        warnings.filterwarnings('ignore')
        
        y_pred = self.best_model.predict(self.X_test_scaled)
        y_pred_proba = self.best_model.predict_proba(self.X_test_scaled)[:, 1]
        
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred, average='weighted')
        test_roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Get CV scores for tuned model
        final_cv = cross_validate(
            self.best_model,
            self.X_train_scaled,
            self.y_train,
            cv=5,
            scoring=['accuracy', 'f1'],
            n_jobs=-1
        )
        
        tuned_cv_f1 = final_cv['test_f1'].mean()
        tuned_cv_acc = final_cv['test_accuracy'].mean()
        
        # Calculate improvements
        base_cv_acc = self.base_cv_results['test_accuracy'].mean()
        f1_improvement = tuned_cv_f1 - base_cv_f1
        acc_improvement = tuned_cv_acc - base_cv_acc
        
        f1_improvement_str = f"(+{f1_improvement:.3f})" if f1_improvement > 0 else f"({f1_improvement:.3f})"
        acc_improvement_str = f"(+{acc_improvement:.3f})" if acc_improvement > 0 else f"({acc_improvement:.3f})"
        
        print(f"\n  Tuned model performance:")
        print(f"    • CV F1 Score: {tuned_cv_f1:.3f} {f1_improvement_str}")
        print(f"    • CV Accuracy: {tuned_cv_acc:.3f} {acc_improvement_str}")
        print(f"    • Test Accuracy: {test_accuracy:.3f}")
        print(f"    • Test AUC: {test_roc_auc:.3f}")
    
    def _display_classification_report(self):
        """Display detailed classification report"""
        y_pred = self.best_model.predict(self.X_test_scaled)
        
        print(f"\n  Classification report:")
        report = classification_report(self.y_test, y_pred, target_names=['Blue', 'Red'], output_dict=True)
        
        print(f"              Precision  Recall  F1-Score  Support")
        for label in ['Blue', 'Red']:
            p = report[label]['precision']
            r = report[label]['recall']
            f1 = report[label]['f1-score']
            s = int(report[label]['support'])
            print(f"    {label:<8}     {p:.2f}     {r:.2f}     {f1:.2f}      {s}")
        
        print(f"    ")
        acc = report['accuracy']
        total_support = int(report['macro avg']['support'])
        print(f"    Overall      {acc:.2f}     {acc:.2f}     {acc:.2f}    {total_support:,}")
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """
        Plot feature importance for the tuned model
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save the plot
        """
        if self.best_model is None:
            print("  ✗ No trained model found. Run tune_model() first.")
            return
        
        if not hasattr(self.best_model, 'feature_importances_'):
            print("  ✗ Model doesn't have feature_importances_ attribute")
            return
        
        feature_imp = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importance': self.best_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_imp, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n  ◦ Feature importance plot saved: {save_path}")
        
        plt.close()
    
    def save_model(self, model_path, scaler_path, columns_path):
        """
        Save trained model, scaler and feature columns
        
        Args:
            model_path: Path to save model
            scaler_path: Path to save scaler
            columns_path: Path to save column names
        """
        if self.best_model is None:
            print("  ✗ No model to save. Run tune_model() first.")
            return
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.X_train.columns.tolist(), columns_path)
        
        print(f"\n  ◦ Model artifacts saved: {model_path.parent}/")
        print(f"    • {model_path.name}")
        print(f"    • {scaler_path.name}")
        print(f"    • {columns_path.name}")
    
    def get_best_model(self):
        """Return the best tuned model"""
        return self.best_model