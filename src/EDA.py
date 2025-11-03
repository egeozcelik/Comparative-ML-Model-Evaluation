import pandas as pd


class EDA:
    """
    Exploratory Data Analysis Module
    Handles data loading, inspection, cleaning and initial analysis
    """
    
    def __init__(self, file_path, columns_to_drop=None):
        """
        Args:
            file_path: Path to the raw dataset
            columns_to_drop: List of columns to remove from dataset
        """
        self.file_path = file_path
        self.columns_to_drop = columns_to_drop or []
        self.df = None
        self.cat_cols = []
        self.num_cols = []
        self.cat_but_car = []
        
    def load_data(self):
        """Load dataset from CSV file"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"  ◦ Data loaded: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            return self.df
        except Exception as e:
            print(f"  ✗ Error loading data: {e}")
            return None
    
    def inspect_data(self):
        """Display basic dataset information"""
        if self.df is None:
            print("  ✗ No data loaded. Run load_data() first.")
            return
        
        if 'winner' in self.df.columns:
            print("\n  Target distribution:")
            value_counts = self.df['winner'].value_counts()
            value_props = self.df['winner'].value_counts(normalize=True)
            for label in value_counts.index:
                count = value_counts[label]
                prop = value_props[label] * 100
                print(f"    • {label}:  {prop:.1f}% ({count:,} samples)")
    
    def analyze_missing_values(self):
        """Analyze and report missing values in dataset"""
        if self.df is None:
            print("  ✗ No data loaded.")
            return None
        
        missing = self.df.isnull().sum()
        missing_cols = missing[missing > 0]
        
        if len(missing_cols) == 0:
            print("\n  ◦ No missing values found")
            return None
        
        missing_df = pd.DataFrame({
            'Column': missing_cols.index,
            'Missing_Count': missing_cols.values,
            'Percentage': (missing_cols.values / len(self.df) * 100).round(2)
        }).sort_values('Percentage', ascending=False)
        
        print(f"\n  Missing values detected in {len(missing_cols)} columns:")
        for _, row in missing_df.iterrows():
            print(f"    • {row['Column']}: {int(row['Missing_Count']):,} missing ({row['Percentage']:.2f}%)")
        
        return missing_df
    
    def drop_columns(self):
        """Remove unnecessary columns from dataset"""
        if self.df is None:
            print("  ✗ No data loaded.")
            return None
        
        existing_cols = [col for col in self.columns_to_drop if col in self.df.columns]
        
        if len(existing_cols) == 0:
            print("\n  ◦ No columns to drop")
            return self.df
        
        # Categorize dropped columns
        fight_outcome = ['method', 'finish_round', 'time_sec']
        event_meta = ['event_name', 'referee']
        per_fight_stats = [col for col in existing_cols if any(x in col for x in ['_sig_str', '_str', '_td', '_sub_att', '_rev', '_ctrl_sec']) and not col.endswith('_diff')]
        redundant_diff = [col for col in existing_cols if col.endswith('_diff')]
        
        print(f"\n  Dropping {len(existing_cols)} unnecessary columns:")
        
        if any(col in existing_cols for col in fight_outcome):
            print(f"    • Fight outcome: {', '.join([c for c in fight_outcome if c in existing_cols])}")
        
        if any(col in existing_cols for col in event_meta):
            print(f"    • Event metadata: {', '.join([c for c in event_meta if c in existing_cols])}")
        
        if per_fight_stats:
            sample_stats = per_fight_stats[:3]
            remaining = len(per_fight_stats) - 3
            stats_str = ', '.join(sample_stats)
            if remaining > 0:
                stats_str += f", ... ({remaining} more)"
            print(f"    • Per-fight stats: {stats_str}")
        
        if redundant_diff:
            sample_diff = redundant_diff[:3]
            remaining = len(redundant_diff) - 3
            diff_str = ', '.join(sample_diff)
            if remaining > 0:
                diff_str += f", ... ({remaining} more)"
            print(f"    • Redundant differences: {diff_str}")
        
        self.df = self.df.drop(columns=existing_cols)
        
        print(f"\n  ◦ Final dataset: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        return self.df
    
    def run_eda(self):
        """Execute complete EDA pipeline"""
        print("\n  Loading dataset...")
        self.load_data()
        if self.df is None:
            return None
        
        self.inspect_data()
        self.analyze_missing_values()
        self.drop_columns()
        
        return self.df
    
    def get_dataframe(self):
        """Return processed dataframe"""
        return self.df