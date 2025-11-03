import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class BaseModelTraining:
    """
    Base Model Training Module
    Handles data preparation, scaling, and baseline model comparison
    """
    
    def __init__(self, dataframe, target_col='winner', test_size=0.2, 
                 random_state=42, model_configs=None):
        """
        Args:
            dataframe: Processed dataframe
            target_col: Target column name
            test_size: Test set proportion
            random_state: Random state for reproducibility
            model_configs: Dictionary of model configurations
        """
        self.df = dataframe.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.model_configs = model_configs or {}
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        
        self.scaler = None
        self.models = {}
        self.results = {}
        
        self.model_map = {
            "Logistic Regression": LogisticRegression,
            "KNN": KNeighborsClassifier,
            "Decision Tree": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier,
            "Gradient Boosting": GradientBoostingClassifier,
            "XGBoost": XGBClassifier,
            "LightGBM": LGBMClassifier,
            "CatBoost": CatBoostClassifier
        }
    
    def prepare_data(self):
        """Split data into train/test sets and apply scaling"""
        cat_cols = self.df.select_dtypes(include=['object']).columns
        cat_cols = [col for col in cat_cols if col != self.target_col]
        
        if len(cat_cols) > 0:
            for col in cat_cols:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
        
        y = self.df[self.target_col]
        X = self.df.drop([self.target_col], axis=1)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"  ◦ Data split: {self.X_train.shape[0]:,} train / {self.X_test.shape[0]:,} test ({int((1-self.test_size)*100)}/{int(self.test_size*100)})")
        
        self.scaler = RobustScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"  ◦ Scaling applied: RobustScaler")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_baseline_models(self):
        """Train multiple baseline models and evaluate performance"""
        import warnings
        warnings.filterwarnings('ignore')
        
        print(f"\n  Training {len(self.model_configs)} baseline models...\n")
        
        for name, config in self.model_configs.items():
            try:
                model_class = self.model_map.get(name)
                if model_class is None:
                    continue
                
                model = model_class(**config)
                model.fit(self.X_train_scaled, self.y_train)
                
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                accuracy = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                self.models[name] = model
                self.results[name] = {
                    'accuracy': accuracy,
                    'f1_score': f1
                }
                
                if y_pred_proba is not None:
                    roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                    self.results[name]['roc_auc'] = roc_auc
                
            except Exception as e:
                pass
    
    def display_results(self):
        """Display model comparison results"""
        if not self.results:
            print("  ✗ No results to display")
            return None
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        print(f"  Results (ranked by F1 score):")
        
        for idx, (name, row) in enumerate(results_df.iterrows(), 1):
            f1 = row['f1_score']
            acc = row['accuracy']
            auc = row.get('roc_auc', 0)
            
            best_marker = "  ✓" if idx == 1 else ""
            print(f"    {idx}. {name:<22} F1: {f1:.3f}  Acc: {acc:.3f}  AUC: {auc:.3f}{best_marker}")
        
        best_model_name = results_df.index[0]
        print(f"\n  ◦ Best performing model: {best_model_name}")
        
        return results_df
    
    def run_base_training(self):
        """Execute complete base model training pipeline"""
        self.prepare_data()
        self.train_baseline_models()
        results_df = self.display_results()
        
        return results_df
    
    def get_best_model_name(self):
        """Return the name of the best performing model"""
        if not self.results:
            return None
        results_df = pd.DataFrame(self.results).T
        return results_df.sort_values('f1_score', ascending=False).index[0]
    
    def get_train_test_data(self):
        """Return train and test data"""
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_scaled_data(self):
        """Return scaled train and test data"""
        return self.X_train_scaled, self.X_test_scaled
    
    def get_scaler(self):
        """Return fitted scaler"""
        return self.scaler