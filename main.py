from pathlib import Path
from src.EDA import EDA
from src.feature_engineering import FeatureEngineering
from src.base_model_training import BaseModelTraining
from src.model_tuning import ModelTuning


class MLPipeline:
    """UFC Fight Prediction - Machine Learning Pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.df_cleaned = None
        self.df_engineered = None
        self.base_trainer = None
        self.tuner = None
        
        self._setup_directories()
    
    def _setup_directories(self):
        for directory in [
            self.config['DATA_DIR'],
            self.config['RAW_DATA_DIR'],
            self.config['PROCESSED_DATA_DIR'],
            self.config['MODELS_DIR'],
            self.config['SAVED_MODELS_DIR']
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def run_eda(self):
        print("\n→ Stage 1: Exploratory Data Analysis")
        
        data_path = self.config['RAW_DATA_DIR'] / self.config['DATASET_NAME']
        eda = EDA(file_path=data_path, columns_to_drop=self.config['COLUMNS_TO_DROP'])
        self.df_cleaned = eda.run_eda()
        
        if self.df_cleaned is None:
            raise ValueError("EDA failed. Check your data path and format.")
        
        return self
    
    def run_feature_engineering(self):
        print("\n→ Stage 2: Feature Engineering")
        
        if self.df_cleaned is None:
            raise ValueError("Run EDA first before feature engineering.")
        
        feature_eng = FeatureEngineering(
            dataframe=self.df_cleaned,
            missing_value_method=self.config['MISSING_VALUE_METHOD'],
            lower_quantile=self.config['LOWER_QUANTILE'],
            upper_quantile=self.config['UPPER_QUANTILE']
        )
        self.df_engineered = feature_eng.run_feature_engineering()
        
        processed_path = self.config['PROCESSED_DATA_DIR'] / "processed_data.csv"
        self.df_engineered.to_csv(processed_path, index=False)
        
        return self
    
    def train_baseline_models(self):
        print("\n→ Stage 3: Baseline Model Training")
        
        if self.df_engineered is None:
            raise ValueError("Run feature engineering first before training models.")
        
        self.base_trainer = BaseModelTraining(
            dataframe=self.df_engineered,
            target_col='winner',
            test_size=self.config['TEST_SIZE'],
            random_state=self.config['RANDOM_STATE'],
            model_configs=self.config['BASELINE_MODELS']
        )
        results_df = self.base_trainer.run_base_training()
        
        return self
    
    def tune_best_model(self):
        print("\n→ Stage 4: Hyperparameter Tuning")
        
        if self.base_trainer is None:
            raise ValueError("Train baseline models first before tuning.")
        
        X_train, X_test, y_train, y_test = self.base_trainer.get_train_test_data()
        X_train_scaled, X_test_scaled = self.base_trainer.get_scaled_data()
        scaler = self.base_trainer.get_scaler()
        best_model_name = self.base_trainer.get_best_model_name()
        
        self.tuner = ModelTuning(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            scaler=scaler,
            model_name=best_model_name,
            param_grid=self.config['PARAM_GRID'],
            search_config=self.config['SEARCH_CONFIG'],
            random_state=self.config['RANDOM_STATE']
        )
        
        best_model = self.tuner.tune_model()
        
        return self
    
    def save_artifacts(self):
        if self.tuner is None:
            raise ValueError("Tune model first before saving artifacts.")
        
        # Save feature importance plot
        plot_path = self.config['MODELS_DIR'] / "feature_importance.png"
        self.tuner.plot_feature_importance(top_n=20, save_path=plot_path)
        
        # Save model artifacts
        self.tuner.save_model(
            model_path=self.config['SAVED_MODELS_DIR'] / "best_model.pkl",
            scaler_path=self.config['SAVED_MODELS_DIR'] / "scaler.pkl",
            columns_path=self.config['SAVED_MODELS_DIR'] / "columns.pkl"
        )
        
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("✓ Pipeline completed successfully")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        
        return self
    
    def run_full_pipeline(self):
        """Execute complete ML pipeline from data to deployment"""
        return (self
                .run_eda()
                .run_feature_engineering()
                .train_baseline_models()
                .tune_best_model()
                .save_artifacts())


def get_config():
    """Pipeline configuration"""
    ROOT_DIR = Path(__file__).parent.absolute()
    DATA_DIR = ROOT_DIR / "data"
    
    return {
        'DATA_DIR': DATA_DIR,
        'RAW_DATA_DIR': DATA_DIR / "raw",
        'PROCESSED_DATA_DIR': DATA_DIR / "processed",
        'MODELS_DIR': ROOT_DIR / "models",
        'SAVED_MODELS_DIR': ROOT_DIR / "models" / "saved_models",
        
        'DATASET_NAME': "large_dataset.csv",
        'RANDOM_STATE': 42,
        'TEST_SIZE': 0.2,
        
        'MISSING_VALUE_METHOD': "median",
        'LOWER_QUANTILE': 0.05,
        'UPPER_QUANTILE': 0.95,
        
        'COLUMNS_TO_DROP': [
            'method', 'finish_round', 'time_sec', 'event_name', 'referee',
            'r_sig_str', 'r_sig_str_att', 'r_sig_str_acc', 'r_str', 'r_str_att', 'r_str_acc',
            'r_td', 'r_td_att', 'r_td_acc', 'r_sub_att', 'r_rev', 'r_ctrl_sec',
            'b_sig_str', 'b_sig_str_att', 'b_sig_str_acc', 'b_str', 'b_str_att', 'b_str_acc',
            'b_td', 'b_td_att', 'b_td_acc', 'b_sub_att', 'b_rev', 'b_ctrl_sec',
            'kd_diff', 'sig_str_diff', 'sig_str_att_diff', 'sig_str_acc_diff',
            'str_diff', 'str_att_diff', 'str_acc_diff',
            'td_diff', 'td_att_diff', 'td_acc_diff',
            'sub_att_diff', 'rev_diff', 'ctrl_sec_diff'
        ],
        
        'BASELINE_MODELS': {
            "Logistic Regression": {"max_iter": 1000, "random_state": 42},
            "KNN": {"n_neighbors": 5},
            "Decision Tree": {"random_state": 42},
            "Random Forest": {"n_estimators": 100, "random_state": 42},
            "Gradient Boosting": {"random_state": 42},
            "XGBoost": {"eval_metric": 'logloss', "random_state": 42, "use_label_encoder": False},
            "LightGBM": {"random_state": 42, "verbose": -1},
            "CatBoost": {"verbose": False, "random_state": 42}
        },
        
        'PARAM_GRID': {
            'iterations': [100, 200, 300],
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5, 7, 9]
        },
        
        'SEARCH_CONFIG': {
            'n_iter': 20,
            'cv': 5,
            'scoring': 'f1',
            'verbose': 1,
            'random_state': 42,
            'n_jobs': -1
        }
    }


if __name__ == "__main__":
    print("UFC Fight Prediction Pipeline")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    config = get_config()
    pipeline = MLPipeline(config)
    
    # Full pipeline execution
    pipeline.run_full_pipeline()
    
    # Or run stages individually:
    # pipeline.run_eda()
    # pipeline.run_feature_engineering()
    # pipeline.train_baseline_models()
    # pipeline.tune_best_model()
    # pipeline.save_artifacts()