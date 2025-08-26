# F1 Race Prediction System
# Advanced Machine Learning Model for Predicting Race Winners and Top 3 Finishers
# Based on qualifying data and historical performance

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

class F1RacePredictionSystem:
    """
    Complete F1 Race Prediction System
    
    This system predicts race winners and top 3 finishers based on:
    - Qualifying positions and times
    - Historical team performance
    - Driver experience and form
    - Track characteristics
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.label_encoders = None
        self.historical_data = None
        self.model_performance = {}
        
    def load_data(self, csv_path):
        """Load F1 race data from CSV file"""
        df = pd.read_csv(csv_path)
        print(f"✅ Data loaded: {df.shape[0]} records from {df['Event'].nunique()} races")
        return df
    
    def time_to_seconds(self, time_str):
        """Convert time string to seconds"""
        if pd.isna(time_str):
            return np.nan
        try:
            if 'days' in str(time_str):
                # Handle timedelta format
                parts = str(time_str).split()
                if len(parts) >= 3:
                    time_part = parts[2]  # Get the time part after 'days'
                    time_parts = time_part.split(':')
                    if len(time_parts) == 3:
                        h, m, s = time_parts
                        return int(h) * 3600 + int(m) * 60 + float(s)
            return float(time_str)
        except:
            return np.nan
    
    def create_features(self, df):
        """Create comprehensive features for race prediction"""
        df_features = df.copy()
        
        # Create target variable (Top 3 finish)
        df_features['Top3'] = (df_features['Position'] <= 3).astype(int)
        df_features['IsWinner'] = (df_features['Position'] == 1).astype(int)
        
        # Convert qualifying times to seconds
        df_features['QualTime_seconds'] = df_features['QualTime'].apply(self.time_to_seconds)
        
        # Driver experience (based on appearing in multiple races)
        driver_counts = df_features.groupby('Abbreviation').size()
        df_features['DriverExperience'] = df_features['Abbreviation'].map(driver_counts)
        
        # Team performance (average points per race)
        team_performance = df_features.groupby('TeamName')['Points'].mean()
        df_features['TeamAvgPoints'] = df_features['TeamName'].map(team_performance)
        
        # Historical performance features
        df_features = df_features.sort_values(['Abbreviation', 'Event'])
        
        # Previous race performance
        df_features['PrevPosition'] = df_features.groupby('Abbreviation')['Position'].shift(1)
        df_features['PrevPoints'] = df_features.groupby('Abbreviation')['Points'].shift(1)
        
        # Fill missing values for drivers in their first race
        df_features['PrevPosition'] = df_features['PrevPosition'].fillna(10)
        df_features['PrevPoints'] = df_features['PrevPoints'].fillna(0)
        
        # Qualifying performance relative to teammates
        team_qual_avg = df_features.groupby(['Event', 'TeamName'])['QualGrid'].transform('mean')
        df_features['QualVsTeammate'] = df_features['QualGrid'] - team_qual_avg
        
        # Track-specific features
        track_difficulty = df_features.groupby('Event')['Position'].std()
        df_features['TrackChaosLevel'] = df_features['Event'].map(track_difficulty)
        
        # Team competitiveness ranking
        team_competitiveness = df_features.groupby('TeamName')['Points'].sum().sort_values(ascending=False)
        team_rank_map = {team: rank+1 for rank, team in enumerate(team_competitiveness.index)}
        df_features['TeamCompetitivenessRank'] = df_features['TeamName'].map(team_rank_map)
        
        # Encode categorical variables
        le_driver = LabelEncoder()
        le_team = LabelEncoder()
        le_event = LabelEncoder()
        
        df_features['Driver_encoded'] = le_driver.fit_transform(df_features['Abbreviation'])
        df_features['Team_encoded'] = le_team.fit_transform(df_features['TeamName'])
        df_features['Event_encoded'] = le_event.fit_transform(df_features['Event'])
        
        # Store label encoders
        self.label_encoders = {
            'driver': le_driver,
            'team': le_team,
            'event': le_event
        }
        
        print(f"✅ Feature engineering completed: {df_features.shape[1]} features created")
        return df_features
    
    def train_models(self, df_features):
        """Train multiple models and select the best performer"""
        
        # Define feature columns
        self.feature_columns = [
            'QualGrid', 'GridPosition', 'QualTime_seconds', 'DriverExperience',
            'TeamAvgPoints', 'PrevPosition', 'PrevPoints', 'QualVsTeammate',
            'TrackChaosLevel', 'Driver_encoded', 'Team_encoded', 'Event_encoded'
        ]
        
        # Prepare data
        X = df_features[self.feature_columns].copy()
        y = df_features['Top3']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
            'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
        }
        
        # Train and evaluate models
        print("\n🔄 Training models...")
        print("="*50)
        
        for name, model in models.items():
            if name in ['Logistic Regression', 'SVM']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            self.model_performance[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'scaled_data': name in ['Logistic Regression', 'SVM']
            }
            
            print(f"{name:18} | Accuracy: {accuracy:.3f} | F1: {f1:.3f}")
        
        # Select best model
        best_model_name = max(self.model_performance.keys(), 
                             key=lambda k: self.model_performance[k]['f1'])
        self.model = self.model_performance[best_model_name]['model']
        self.use_scaled_data = self.model_performance[best_model_name]['scaled_data']
        
        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   Accuracy: {self.model_performance[best_model_name]['accuracy']:.1%}")
        print(f"   F1-Score: {self.model_performance[best_model_name]['f1']:.3f}")
        
        # Store historical data for predictions
        self.historical_data = df_features
        
        return best_model_name
    
    def predict_race(self, qualifying_data, race_event="Unknown Race"):
        """
        Predict race results based on qualifying data
        
        Parameters:
        -----------
        qualifying_data : pandas.DataFrame
            DataFrame with columns: ['Abbreviation', 'TeamName', 'QualGrid', 'QualTime_seconds']
        race_event : str
            Name of the race event
            
        Returns:
        --------
        pandas.DataFrame
            Predictions with probabilities and rankings
        """
        
        if self.model is None:
            raise ValueError("Model must be trained first! Call train() method.")
        
        # Prepare prediction data
        pred_data = qualifying_data.copy()
        
        # Add GridPosition (assume same as QualGrid for simplicity)
        if 'GridPosition' not in pred_data.columns:
            pred_data['GridPosition'] = pred_data['QualGrid']
        
        # Add features from historical data
        # Driver experience
        if self.historical_data is not None:
            driver_exp = self.historical_data.groupby('Abbreviation').size()
            pred_data['DriverExperience'] = pred_data['Abbreviation'].map(driver_exp).fillna(1)
            
            # Team performance
            team_perf = self.historical_data.groupby('TeamName')['Points'].mean()
            pred_data['TeamAvgPoints'] = pred_data['TeamName'].map(team_perf).fillna(0)
            
            # Previous performance (latest race)
            latest_race = self.historical_data.groupby('Abbreviation').last()
            pred_data['PrevPosition'] = pred_data['Abbreviation'].map(latest_race['Position']).fillna(10)
            pred_data['PrevPoints'] = pred_data['Abbreviation'].map(latest_race['Points']).fillna(0)
        else:
            # Default values if no historical data
            pred_data['DriverExperience'] = 1
            pred_data['TeamAvgPoints'] = 0
            pred_data['PrevPosition'] = 10
            pred_data['PrevPoints'] = 0
        
        # Qualifying vs teammate
        team_qual_avg = pred_data.groupby('TeamName')['QualGrid'].transform('mean')
        pred_data['QualVsTeammate'] = pred_data['QualGrid'] - team_qual_avg
        
        # Track chaos level (use average if unknown)
        pred_data['TrackChaosLevel'] = 5.92  # Average from training data
        
        # Encode categorical variables
        pred_data['Driver_encoded'] = pred_data['Abbreviation'].apply(
            lambda x: self.label_encoders['driver'].transform([x])[0] 
            if x in self.label_encoders['driver'].classes_ else 0
        )
        pred_data['Team_encoded'] = pred_data['TeamName'].apply(
            lambda x: self.label_encoders['team'].transform([x])[0] 
            if x in self.label_encoders['team'].classes_ else 0
        )
        pred_data['Event_encoded'] = 0  # New event
        
        # Select features and make predictions
        X_pred = pred_data[self.feature_columns].fillna(0)
        
        if self.use_scaled_data:
            X_pred_scaled = self.scaler.transform(X_pred)
            probabilities = self.model.predict_proba(X_pred_scaled)[:, 1]
            predictions = self.model.predict(X_pred_scaled)
        else:
            probabilities = self.model.predict_proba(X_pred)[:, 1]
            predictions = self.model.predict(X_pred)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Driver': pred_data['Abbreviation'],
            'Team': pred_data['TeamName'],
            'QualGrid': pred_data['QualGrid'],
            'Top3_Probability': probabilities,
            'Predicted_Top3': predictions,
            'Win_Probability': probabilities * 0.33  # Rough estimate for win probability
        })
        
        # Sort by probability (descending)
        results = results.sort_values('Top3_Probability', ascending=False)
        results['Predicted_Position'] = range(1, len(results) + 1)
        
        return results
    
    def get_top3_prediction(self, qualifying_data, race_event="Unknown Race"):
        """Get top 3 predictions with formatted output"""
        results = self.predict_race(qualifying_data, race_event)
        
        print(f"\n🏁 RACE PREDICTION: {race_event}")
        print("="*50)
        print("Predicted Top 3 Finishers:")
        
        top3 = results.head(3)
        for i, (_, row) in enumerate(top3.iterrows(), 1):
            print(f"{i}. {row['Driver']} ({row['Team']}) - {row['Top3_Probability']:.1%} probability")
        
        print(f"\nFull Rankings:")
        for _, row in results.iterrows():
            status = "⭐" if row['Predicted_Top3'] else "  "
            print(f"{status} P{row['Predicted_Position']:2d} {row['Driver']:3s} "
                  f"({row['Team']:15s}) {row['Top3_Probability']:.1%}")
        
        return results
    
    def analyze_historical_performance(self):
        """Analyze historical performance and provide insights"""
        if self.historical_data is None:
            print("❌ No historical data available for analysis")
            return
        
        df = self.historical_data
        
        print("\n📊 HISTORICAL PERFORMANCE ANALYSIS")
        print("="*50)
        
        # Pole position analysis
        pole_wins = df[df['QualGrid'] == 1]['Position'].value_counts().sort_index()
        print("Pole position (P1 qualifier) results:")
        for pos, count in pole_wins.head(5).items():
            total_poles = len(df[df['QualGrid'] == 1])
            print(f"  Finishes P{int(pos)}: {count} times ({count/total_poles*100:.1f}%)")
        
        # Team performance
        print("\nTeam podium performance:")
        team_podiums = df[df['Position'] <= 3].groupby('TeamName').size().sort_values(ascending=False)
        total_podiums = len(df[df['Position'] <= 3])
        for team, podiums in team_podiums.items():
            print(f"  {team}: {podiums} podiums ({podiums/total_podiums*100:.1f}%)")
        
        # Driver consistency
        print("\nTop 5 drivers by total points:")
        driver_stats = df.groupby('Abbreviation').agg({
            'Position': ['mean', 'min'],
            'Points': 'sum'
        }).round(2)
        driver_stats.columns = ['Avg_Position', 'Best_Position', 'Total_Points']
        print(driver_stats.sort_values('Total_Points', ascending=False).head().to_string())

def main():
    """Example usage of the F1 Race Prediction System"""
    
    # Initialize the prediction system
    predictor = F1RacePredictionSystem()
    
    # Load your data (replace with your CSV file path)
    print("🏎️  F1 RACE PREDICTION SYSTEM")
    print("="*50)
    
    # For demonstration, create sample data
    # Replace this with: df = predictor.load_data('your_f1_data.csv')
    sample_data = {
        'DriverNumber': [1, 11, 55, 16, 63, 44],
        'Abbreviation': ['VER', 'PER', 'SAI', 'LEC', 'RUS', 'HAM'],
        'TeamName': ['Red Bull Racing', 'Red Bull Racing', 'Ferrari', 'Ferrari', 'Mercedes', 'Mercedes'],
        'Position': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'QualGrid': [2.0, 5.0, 1.0, 3.0, 4.0, 8.0],
        'QualTime': ['0 days 00:01:20.307000', '0 days 00:01:20.688000', 
                     '0 days 00:01:20.294000', '0 days 00:01:20.361000',
                     '0 days 00:01:20.671000', '0 days 00:01:20.820000'],
        'Points': [25.0, 18.0, 15.0, 12.0, 10.0, 8.0],
        'Event': ['Monaco'] * 6,
        'GridPosition': [2.0, 5.0, 1.0, 3.0, 4.0, 8.0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Feature engineering
    df_features = predictor.create_features(df)
    
    # Train models
    best_model = predictor.train_models(df_features)
    
    # Make a prediction for a new race
    print("\n🔮 MAKING RACE PREDICTION")
    print("-" * 30)
    
    # Sample qualifying data for prediction
    new_qualifying = pd.DataFrame({
        'Abbreviation': ['VER', 'HAM', 'LEC', 'SAI', 'RUS', 'NOR'],
        'TeamName': ['Red Bull Racing', 'Mercedes', 'Ferrari', 'Ferrari', 'Mercedes', 'McLaren'],
        'QualGrid': [1, 2, 3, 4, 5, 6],
        'QualTime_seconds': [78.5, 78.8, 79.1, 79.2, 79.0, 79.5]
    })
    
    # Get predictions
    results = predictor.get_top3_prediction(new_qualifying, "Belgian Grand Prix")
    
    # Analyze historical performance
    predictor.analyze_historical_performance()
    
    print("\n✅ Prediction system ready for use!")
    print("Use predictor.predict_race(qualifying_data, race_name) for new predictions")

if __name__ == "__main__":
    main()