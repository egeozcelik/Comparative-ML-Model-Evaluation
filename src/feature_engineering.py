import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class FeatureEngineering:
    """
    Feature Engineering Module
    Creates new features, handles missing values, and processes outliers
    """
    
    def __init__(self, dataframe, missing_value_method="median", 
                 lower_quantile=0.05, upper_quantile=0.95):
        """
        Args:
            dataframe: Input dataframe
            missing_value_method: Method for filling numerical missing values
            lower_quantile: Lower quantile for outlier capping
            upper_quantile: Upper quantile for outlier capping
        """
        self.df = dataframe.copy()
        self.missing_value_method = missing_value_method
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.label_encoders = {}
    
    def create_bmi_features(self):
        """Calculate Body Mass Index for both fighters"""
        self.df["r_body_mass_index"] = self.df["r_weight"] / ((self.df["r_height"] / 100) ** 2)
        self.df["b_body_mass_index"] = self.df["b_weight"] / ((self.df["b_height"] / 100) ** 2)
        
        return ['r_body_mass_index', 'b_body_mass_index']
    
    def create_difference_features(self):
        """Create difference features between red and blue corners"""
        features = [
            'wins_total_diff', 'losses_total_diff',
            'age_diff', 'height_diff', 'weight_diff', 'reach_diff',
            'SLpM_total_diff', 'SApM_total_diff',
            'sig_str_acc_total_diff', 'td_acc_total_diff',
            'str_def_total_diff', 'td_def_total_diff',
            'sub_avg_diff', 'td_avg_diff', 'kd_diff'
        ]
        
        self.df["wins_total_diff"] = self.df["r_wins_total"] - self.df["b_wins_total"]
        self.df["losses_total_diff"] = self.df["r_losses_total"] - self.df["b_losses_total"]
        self.df["age_diff"] = self.df["r_age"] - self.df["b_age"]
        self.df["height_diff"] = self.df["r_height"] - self.df["b_height"]
        self.df["weight_diff"] = self.df["r_weight"] - self.df["b_weight"]
        self.df["reach_diff"] = self.df["r_reach"] - self.df["b_reach"]
        
        self.df["SLpM_total_diff"] = self.df["r_SLpM_total"] - self.df["b_SLpM_total"]
        self.df["SApM_total_diff"] = self.df["r_SApM_total"] - self.df["b_SApM_total"]
        self.df["sig_str_acc_total_diff"] = self.df["r_sig_str_acc_total"] - self.df["b_sig_str_acc_total"]
        self.df["td_acc_total_diff"] = self.df["r_td_acc_total"] - self.df["b_td_acc_total"]
        self.df["str_def_total_diff"] = self.df["r_str_def_total"] - self.df["b_str_def_total"]
        self.df["td_def_total_diff"] = self.df["r_td_def_total"] - self.df["b_td_def_total"]
        self.df["sub_avg_diff"] = self.df["r_sub_avg"] - self.df["b_sub_avg"]
        self.df["td_avg_diff"] = self.df["r_td_avg"] - self.df["b_td_avg"]
        self.df["kd_diff"] = self.df["r_kd"] - self.df["b_kd"]
        
        return features
    
    def create_comparative_features(self):
        """Create binary comparison features"""
        features = [
            'is_r_younger', 'is_r_taller', 'is_r_more_experienced',
            'is_r_better_td_acc', 'same_stance'
        ]
        
        self.df["is_r_younger"] = (self.df["r_age"] < self.df["b_age"]).astype(int)
        self.df["is_r_taller"] = (self.df["r_height"] > self.df["b_height"]).astype(int)
        self.df["is_r_more_experienced"] = (self.df["r_wins_total"] > self.df["b_wins_total"]).astype(int)
        self.df["is_r_better_td_acc"] = (self.df["r_td_acc_total"] > self.df["b_td_acc_total"]).astype(int)
        self.df["same_stance"] = (self.df["r_stance"] == self.df["b_stance"]).astype(int)
        
        return features
    
    def create_power_features(self):
        """Create power and defense score features"""
        features = [
            'r_power_striking_score', 'b_power_striking_score',
            'r_defense_score', 'b_defense_score',
            'overall_score_diff'
        ]
        
        self.df["r_power_striking_score"] = self.df["r_SLpM_total"] * self.df["r_sig_str_acc_total"]
        self.df["b_power_striking_score"] = self.df["b_SLpM_total"] * self.df["b_sig_str_acc_total"]
        
        self.df["r_defense_score"] = self.df["r_str_def_total"] + self.df["r_td_def_total"]
        self.df["b_defense_score"] = self.df["b_str_def_total"] + self.df["b_td_def_total"]
        
        self.df["overall_score_diff"] = (
            (self.df["r_SLpM_total"] - self.df["b_SLpM_total"]) * 0.4 +
            (self.df["r_str_def_total"] - self.df["b_str_def_total"]) * 0.3 +
            (self.df["r_td_def_total"] - self.df["b_td_def_total"]) * 0.3
        )
        
        return features
    
    def handle_missing_values(self):
        """Fill missing values using specified method"""
        before_missing = self.df.isnull().sum().sum()
        
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if self.df[col].isnull().sum() > 0:
                mode_value = self.df[col].mode()
                if len(mode_value) > 0:
                    self.df[col].fillna(mode_value[0], inplace=True)
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if self.df[col].isnull().sum() > 0:
                if self.missing_value_method == "median":
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
        
        after_missing = self.df.isnull().sum().sum()
        
        return before_missing, after_missing
    
    def handle_outliers(self):
        """Cap outliers using quantile method"""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        num_cols = [col for col in num_cols if 'winner' not in col]
        
        outlier_count = 0
        
        for col in num_cols:
            lower = self.df[col].quantile(self.lower_quantile)
            upper = self.df[col].quantile(self.upper_quantile)
            
            outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            outlier_count += outliers
            
            self.df[col] = np.where(self.df[col] < lower, lower, self.df[col])
            self.df[col] = np.where(self.df[col] > upper, upper, self.df[col])
        
        return outlier_count
    
    def encode_target(self):
        """Encode target variable"""
        if 'winner' in self.df.columns:
            le = LabelEncoder()
            self.df['winner'] = le.fit_transform(self.df['winner'])
            self.label_encoders['winner'] = le
            return dict(zip(le.classes_, le.transform(le.classes_)))
        return None
    
    def drop_physical_features(self):
        """Drop original physical features after creating differences"""
        cols_to_drop = ["r_height", "r_weight", "b_height", "b_weight", "r_reach", "b_reach"]
        existing_cols = [col for col in cols_to_drop if col in self.df.columns]
        
        self.df.drop(columns=existing_cols, inplace=True)
        
        return existing_cols
    
    def run_feature_engineering(self):
        """Execute complete feature engineering pipeline"""
        print(f"\n  Creating new features...\n")
        
        # BMI features
        bmi_features = self.create_bmi_features()
        print(f"  BMI features ({len(bmi_features)}):")
        print(f"    • r_body_mass_index: Red corner BMI")
        print(f"    • b_body_mass_index: Blue corner BMI")
        
        # Difference features
        diff_features = self.create_difference_features()
        print(f"\n  Difference features ({len(diff_features)}):")
        print(f"    • wins_total_diff, losses_total_diff")
        print(f"    • age_diff, height_diff, weight_diff, reach_diff")
        print(f"    • SLpM_total_diff, SApM_total_diff")
        print(f"    • sig_str_acc_total_diff, td_acc_total_diff")
        print(f"    • str_def_total_diff, td_def_total_diff")
        print(f"    • sub_avg_diff, td_avg_diff, kd_diff")
        
        # Comparative features
        comp_features = self.create_comparative_features()
        print(f"\n  Comparative features ({len(comp_features)}):")
        print(f"    • is_r_younger: Red fighter is younger")
        print(f"    • is_r_taller: Red fighter is taller")
        print(f"    • is_r_more_experienced: Red fighter has more wins")
        print(f"    • is_r_better_td_acc: Red fighter has better takedown accuracy")
        print(f"    • same_stance: Both fighters have same stance")
        
        # Power features
        power_features = self.create_power_features()
        print(f"\n  Power & defense features ({len(power_features)}):")
        print(f"    • r_power_striking_score, b_power_striking_score")
        print(f"    • r_defense_score, b_defense_score")
        print(f"    • overall_score_diff: Combined performance differential")
        
        # Data preprocessing
        print(f"\n  Data preprocessing...")
        
        before_missing, after_missing = self.handle_missing_values()
        method_text = "median for numeric, mode for categorical"
        print(f"  ◦ Missing values: {before_missing:,} → {after_missing} ({method_text})")
        
        outlier_count = self.handle_outliers()
        percentile_range = f"{int(self.lower_quantile*100)}th-{int(self.upper_quantile*100)}th percentile range"
        print(f"  ◦ Outliers capped: {outlier_count:,} values ({percentile_range})")
        
        encoding_map = self.encode_target()
        if encoding_map:
            encoding_str = ", ".join([f"{k}={v}" for k, v in encoding_map.items()])
            print(f"  ◦ Target encoded: {encoding_str}")
        
        dropped_features = self.drop_physical_features()
        print(f"  ◦ Dropped redundant features: {', '.join(dropped_features)}")
        
        print(f"\n  ◦ Feature engineering complete: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        return self.df
    
    def get_dataframe(self):
        """Return engineered dataframe"""
        return self.df