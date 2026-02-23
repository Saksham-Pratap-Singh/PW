# ==================== CRYPTOCURRENCY VOLATILITY PREDICTION ====================
# Complete Python Implementation - Ready to Run
# ==================================================================================

# ==================== 1. DATA PREPROCESSING & FEATURE ENGINEERING ====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class CryptoVolatilityPredictor:
    """Complete pipeline for crypto volatility prediction"""
    
    def __init__(self, filepath=None):
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        if filepath:
            self.load_data(filepath)
    
    # ======================== DATA LOADING & CLEANING ========================
    
    def load_data(self, filepath):
        """Load cryptocurrency data"""
        print("📥 Loading cryptocurrency data...")
        self.df = pd.read_csv(filepath)
        print(f"✓ Loaded {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self
    
    def handle_missing_values(self):
        """Handle missing values with time-series awareness"""
        print("\n🔧 Handling missing values...")
        
        # Forward fill for time-series data
        self.df['open'] = self.df.groupby('symbol')['open'].fillna(method='ffill')
        self.df['high'] = self.df.groupby('symbol')['high'].fillna(method='ffill')
        self.df['low'] = self.df.groupby('symbol')['low'].fillna(method='ffill')
        self.df['close'] = self.df.groupby('symbol')['close'].fillna(method='ffill')
        
        # Backward fill
        self.df[['open', 'high', 'low', 'close']] = \
            self.df[['open', 'high', 'low', 'close']].fillna(method='bfill')
        
        # Volume fill with median
        self.df['volume'] = self.df.groupby('symbol')['volume'].fillna(
            self.df.groupby('symbol')['volume'].median()
        )
        
        # Market cap
        self.df['market_cap'] = self.df.groupby('symbol')['market_cap'].fillna(method='ffill')
        self.df['market_cap'] = self.df['market_cap'].fillna(method='bfill')
        
        # Drop remaining nulls
        self.df = self.df.dropna()
        
        print(f"✓ Missing values handled. Remaining shape: {self.df.shape}")
        return self
    
    def remove_outliers(self, columns=['close', 'volume'], k=1.5):
        """Remove outliers using IQR method"""
        print("\n🎯 Removing outliers...")
        
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR
            
            self.df = self.df[(self.df[col] >= lower_bound) & 
                             (self.df[col] <= upper_bound)]
        
        print(f"✓ Outliers removed. New shape: {self.df.shape}")
        return self
    
    # ======================== FEATURE ENGINEERING ========================
    
    def add_returns(self):
        """Calculate daily returns"""
        print("\n📊 Adding return features...")
        
        self.df['daily_return'] = self.df.groupby('symbol')['close'].pct_change()
        self.df['log_return'] = self.df.groupby('symbol')['close'].apply(
            lambda x: np.log(x / x.shift(1))
        )
        
        print("✓ Return features added")
        return self
    
    def add_volatility_features(self, windows=[5, 10, 20, 50]):
        """Add rolling volatility metrics"""
        print("📈 Adding volatility features...")
        
        for window in windows:
            self.df[f'volatility_{window}'] = self.df.groupby('symbol')[
                'log_return'
            ].rolling(window).std().reset_index(0, drop=True)
        
        # Parkinson volatility
        self.df['parkinson_vol'] = self.df.groupby('symbol').apply(
            lambda x: np.sqrt(
                np.log(x['high'] / x['low']) ** 2 / (4 * np.log(2))
            )
        ).reset_index(0, drop=True)
        
        print("✓ Volatility features added")
        return self
    
    def add_moving_averages(self, windows=[5, 10, 20, 50, 200]):
        """Add moving average features"""
        print("📉 Adding moving average features...")
        
        for window in windows:
            self.df[f'sma_{window}'] = self.df.groupby('symbol')[
                'close'
            ].rolling(window).mean().reset_index(0, drop=True)
        
        # EMA
        for window in [12, 26]:
            self.df[f'ema_{window}'] = self.df.groupby('symbol')[
                'close'
            ].ewm(span=window, adjust=False).mean().reset_index(0, drop=True)
        
        print("✓ Moving average features added")
        return self
    
    def add_bollinger_bands(self, window=20, num_std=2):
        """Add Bollinger Bands"""
        print("🎯 Adding Bollinger Bands...")
        
        def calc_bb(group):
            sma = group['close'].rolling(window).mean()
            std = group['close'].rolling(window).std()
            group[f'bb_upper'] = sma + (std * num_std)
            group[f'bb_lower'] = sma - (std * num_std)
            group[f'bb_width'] = group[f'bb_upper'] - group[f'bb_lower']
            return group
        
        self.df = self.df.groupby('symbol', group_keys=False).apply(calc_bb)
        print("✓ Bollinger Bands added")
        return self
    
    def add_atr(self, period=14):
        """Add Average True Range"""
        print("💹 Adding ATR...")
        
        def calc_atr(group):
            high_low = group['high'] - group['low']
            high_close = abs(group['high'] - group['close'].shift())
            low_close = abs(group['low'] - group['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            group[f'atr'] = true_range.rolling(period).mean()
            return group
        
        self.df = self.df.groupby('symbol', group_keys=False).apply(calc_atr)
        print("✓ ATR added")
        return self
    
    def add_rsi(self, period=14):
        """Add Relative Strength Index"""
        print("📊 Adding RSI...")
        
        def calc_rsi(group):
            delta = group['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            group['rsi'] = 100 - (100 / (1 + rs))
            return group
        
        self.df = self.df.groupby('symbol', group_keys=False).apply(calc_rsi)
        print("✓ RSI added")
        return self
    
    def add_price_features(self):
        """Add price-based features"""
        print("💰 Adding price features...")
        
        self.df['price_range'] = (self.df['high'] - self.df['low']) / self.df['close']
        self.df['price_change'] = (self.df['close'] - self.df['open']) / self.df['open']
        self.df['hl_ratio'] = self.df['high'] / self.df['low']
        
        print("✓ Price features added")
        return self
    
    def add_liquidity_features(self):
        """Add liquidity features"""
        print("💧 Adding liquidity features...")
        
        self.df['dollar_volume'] = self.df['close'] * self.df['volume']
        self.df['vol_mcap_ratio'] = self.df['volume'] / (self.df['market_cap'] + 1e-8)
        
        print("✓ Liquidity features added")
        return self
    
    def create_target_variable(self, forward_window=20):
        """Create target variable for prediction"""
        print("\n🎯 Creating target variable...")
        
        # Future volatility (regression target)
        self.df['future_volatility'] = self.df.groupby('symbol')[
            'log_return'
        ].rolling(forward_window).std().shift(-forward_window).reset_index(0, drop=True)
        
        # Volatility classification
        self.df['volatility_class'] = pd.qcut(
            self.df['future_volatility'],
            q=3,
            labels=['Low', 'Medium', 'High'],
            duplicates='drop'
        )
        
        print(f"✓ Target variable created")
        print(f"  Distribution: {self.df['volatility_class'].value_counts().to_dict()}")
        return self
    
    # ======================== PREPROCESSING PIPELINE ========================
    
    def preprocess_pipeline(self):
        """Run complete preprocessing pipeline"""
        print("=" * 80)
        print("STARTING PREPROCESSING PIPELINE")
        print("=" * 80)
        
        (self
         .handle_missing_values()
         .remove_outliers()
         .add_returns()
         .add_volatility_features()
         .add_moving_averages()
         .add_bollinger_bands()
         .add_atr()
         .add_rsi()
         .add_price_features()
         .add_liquidity_features()
         .create_target_variable())
        
        # Drop NaN values
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        dropped = initial_rows - len(self.df)
        
        print(f"\n✓ Dropped {dropped} rows with NaN values")
        print(f"✓ Final dataset shape: {self.df.shape}")
        print("=" * 80)
        
        return self
    
    # ======================== EDA & VISUALIZATION ========================
    
    def exploratory_analysis(self, output_dir='visualizations/'):
        """Perform exploratory data analysis"""
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistical summary
        print("\n📊 Statistical Summary:")
        print(self.df.describe())
        
        # Correlation analysis
        print("\n🔗 Correlation with target variable:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlations = self.df[numeric_cols].corr()['future_volatility'].sort_values(ascending=False)
        print(correlations.head(15))
        
        # Distribution plot
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(self.df['log_return'].dropna(), bins=50, edgecolor='black')
        plt.title('Distribution of Log Returns')
        plt.xlabel('Log Return')
        
        plt.subplot(1, 3, 2)
        plt.hist(self.df['volume'], bins=50, edgecolor='black')
        plt.title('Distribution of Volume')
        plt.xlabel('Volume')
        
        plt.subplot(1, 3, 3)
        plt.hist(self.df['future_volatility'].dropna(), bins=50, edgecolor='black')
        plt.title('Distribution of Future Volatility (Target)')
        plt.xlabel('Volatility')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}distribution_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualizations saved to {output_dir}")
        
        return self
    
    # ======================== MODEL DEVELOPMENT ========================
    
    def prepare_data(self, test_size=0.2):
        """Prepare data for modeling"""
        print("\n" + "=" * 80)
        print("PREPARING DATA FOR MODELING")
        print("=" * 80)
        
        # Select features (exclude identifiers and target)
        exclude_cols = ['symbol', 'date', 'future_volatility', 'volatility_class', 
                       'daily_return', 'log_return']  # Exclude raw returns
        
        X = self.df.drop([col for col in exclude_cols if col in self.df.columns], 
                        axis=1, errors='ignore')
        y = self.df['future_volatility']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        print(f"✓ Number of features: {len(self.feature_names)}")
        print(f"✓ Features: {self.feature_names[:10]}... (showing first 10)")
        
        # Train-test split (time-series aware)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\n✓ Train set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train XGBoost model"""
        print("\n" + "=" * 80)
        print("TRAINING MODEL")
        print("=" * 80)
        
        print("\n🤖 Building XGBoost model...")
        
        self.model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        print("⏳ Training in progress...")
        self.model.fit(X_train, y_train, verbose=False)
        
        print("✓ Model trained successfully!")
        return self
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        
        y_pred = self.model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2,
            'MAPE': mape
        }
        
        print("\n📊 Model Performance Metrics:")
        print(f"  RMSE:      {rmse:.6f} (Avg prediction error: {rmse*100:.2f}%)")
        print(f"  MAE:       {mae:.6f} (Median error: {mae*100:.2f}%)")
        print(f"  R² Score:  {r2:.4f} (Explains {r2*100:.2f}% of variance)")
        print(f"  MAPE:      {mape:.2f}%")
        
        return metrics, y_pred
    
    def feature_importance(self, top_n=15):
        """Get feature importance"""
        print("\n" + "=" * 80)
        print(f"TOP {top_n} IMPORTANT FEATURES")
        print("=" * 80)
        
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Feature Importance Ranking:")
        for idx, row in importances.head(top_n).iterrows():
            print(f"  {idx+1:2d}. {row['feature']:25s} → {row['importance']:.4f}")
        
        return importances
    
    def save_model(self, filepath='volatility_model.pkl'):
        """Save trained model"""
        print(f"\n💾 Saving model to {filepath}...")
        joblib.dump((self.model, self.scaler, self.feature_names), filepath)
        print("✓ Model saved successfully!")
        return self
    
    @staticmethod
    def load_model(filepath='volatility_model.pkl'):
        """Load trained model"""
        model, scaler, feature_names = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
        return model, scaler, feature_names
    
    def predict(self, X, use_scaler=True):
        """Make predictions"""
        if use_scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "🚀" * 40)
    print("CRYPTOCURRENCY VOLATILITY PREDICTION SYSTEM")
    print("🚀" * 40 + "\n")
    
    # 1. Initialize and preprocess
    predictor = CryptoVolatilityPredictor()
    
    # Load your data
    # predictor.load_data('cryptocurrency_data.csv')
    # predictor.preprocess_pipeline()
    
    # 2. EDA
    # predictor.exploratory_analysis()
    
    # 3. Model training
    # X_train, X_test, y_train, y_test = predictor.prepare_data()
    # predictor.train_model(X_train, y_train)
    
    # 4. Evaluation
    # metrics, y_pred = predictor.evaluate_model(X_test, y_test)
    # predictor.feature_importance()
    
    # 5. Save model
    # predictor.save_model('volatility_model.pkl')
    
    print("\n✓ Setup complete! Use the above commented code with your data.")
    print("\n" + "="*80)
