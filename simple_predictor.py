# Simple F1 Race Predictor
# Quick and easy race winner prediction

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

def predict_f1_race_simple(csv_file_path, qualifying_data):
    """
    Simple F1 race prediction function
    
    Parameters:
    -----------
    csv_file_path : str
        Path to your F1 CSV data file
    qualifying_data : dict
        Dictionary with keys: 'drivers', 'teams', 'grid_positions', 'qual_times'
        Example: {
            'drivers': ['VER', 'HAM', 'LEC'],
            'teams': ['Red Bull Racing', 'Mercedes', 'Ferrari'],
            'grid_positions': [1, 2, 3],
            'qual_times': [78.5, 78.8, 79.1]
        }
    
    Returns:
    --------
    list : Top 3 predicted finishers with probabilities
    """
    
    print("🏎️ F1 Race Predictor")
    print("=" * 25)
    
    # Load data
    df = pd.read_csv(csv_file_path)
    print(f"✅ Loaded {len(df)} records")
    
    # Basic feature engineering
    df['Top3'] = (df['Position'] <= 3).astype(int)
    
    # Convert qualifying times
    def time_to_seconds(time_str):
        if pd.isna(time_str):
            return np.nan
        try:
            if 'days' in str(time_str):
                parts = str(time_str).split()
                if len(parts) >= 3:
                    time_part = parts[2]
                    time_parts = time_part.split(':')
                    if len(time_parts) == 3:
                        h, m, s = time_parts
                        return int(h) * 3600 + int(m) * 60 + float(s)
            return float(time_str)
        except:
            return np.nan
    
    df['QualTime_seconds'] = df['QualTime'].apply(time_to_seconds)
    
    # Team performance
    team_performance = df.groupby('TeamName')['Points'].mean()
    df['TeamAvgPoints'] = df['TeamName'].map(team_performance)
    
    # Encode categorical variables
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    
    df['Driver_encoded'] = le_driver.fit_transform(df['Abbreviation'])
    df['Team_encoded'] = le_team.fit_transform(df['TeamName'])
    
    # Features for model
    features = ['QualGrid', 'GridPosition', 'QualTime_seconds', 'TeamAvgPoints', 'Driver_encoded', 'Team_encoded']
    
    X = df[features].fillna(df[features].mean())
    y = df['Top3']
    
    # Train model
    model = LogisticRegression(random_state=42, class_weight='balanced')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    print("✅ Model trained")
    
    # Prepare prediction data
    pred_df = pd.DataFrame({
        'Abbreviation': qualifying_data['drivers'],
        'TeamName': qualifying_data['teams'],
        'QualGrid': qualifying_data['grid_positions'],
        'QualTime_seconds': qualifying_data['qual_times']
    })
    
    pred_df['GridPosition'] = pred_df['QualGrid']
    pred_df['TeamAvgPoints'] = pred_df['TeamName'].map(team_performance).fillna(0)
    
    # Handle new drivers/teams
    pred_df['Driver_encoded'] = pred_df['Abbreviation'].apply(
        lambda x: le_driver.transform([x])[0] if x in le_driver.classes_ else 0
    )
    pred_df['Team_encoded'] = pred_df['TeamName'].apply(
        lambda x: le_team.transform([x])[0] if x in le_team.classes_ else 0
    )
    
    X_pred = pred_df[features].fillna(0)
    X_pred_scaled = scaler.transform(X_pred)
    
    # Make predictions
    probabilities = model.predict_proba(X_pred_scaled)[:, 1]
    
    # Create results
    results = pd.DataFrame({
        'Driver': pred_df['Abbreviation'],
        'Team': pred_df['TeamName'],
        'Grid': pred_df['QualGrid'],
        'Top3_Probability': probabilities
    })
    
    results = results.sort_values('Top3_Probability', ascending=False)
    
    print("\n🏁 RACE PREDICTIONS:")
    print("-" * 20)
    
    top3_predictions = []
    for i, (_, row) in enumerate(results.head(3).iterrows(), 1):
        prediction = f"P{i}: {row['Driver']} ({row['Team']}) - {row['Top3_Probability']:.1%}"
        print(prediction)
        top3_predictions.append({
            'position': i,
            'driver': row['Driver'],
            'team': row['Team'],
            'probability': row['Top3_Probability']
        })
    
    return top3_predictions

# Example usage
if __name__ == "__main__":
    
    # Example qualifying data for next race
    next_race_qualifying = {
        'drivers': ['VER', 'HAM', 'LEC', 'SAI', 'RUS', 'NOR', 'PER', 'ALO'],
        'teams': ['Red Bull Racing', 'Mercedes', 'Ferrari', 'Ferrari', 
                 'Mercedes', 'McLaren', 'Red Bull Racing', 'Aston Martin'],
        'grid_positions': [1, 2, 3, 4, 5, 6, 7, 8],
        'qual_times': [78.5, 78.8, 79.1, 79.2, 79.0, 79.5, 78.9, 79.7]
    }
    
    # Make prediction (replace with your CSV file path)
    predictions = predict_f1_race_simple('f1_race_dataset_initial.csv', next_race_qualifying)
    
    print(f"\n🎯 Model predicts:")
    print(f"   Winner: {predictions[0]['driver']} ({predictions[0]['probability']:.1%} chance)")
    print(f"   Podium: {[p['driver'] for p in predictions]}")

# Quick prediction function for interactive use
def quick_predict():
    """Interactive prediction function"""
    print("🔮 Quick F1 Race Prediction")
    print("Enter qualifying data:")
    
    drivers = input("Drivers (comma-separated, e.g., VER,HAM,LEC): ").split(',')
    teams = input("Teams (comma-separated): ").split(',')
    positions = list(map(int, input("Grid positions (e.g., 1,2,3): ").split(',')))
    times = list(map(float, input("Qualifying times in seconds (e.g., 78.5,78.8,79.1): ").split(',')))
    
    qualifying_data = {
        'drivers': [d.strip().upper() for d in drivers],
        'teams': [t.strip() for t in teams],
        'grid_positions': positions,
        'qual_times': times
    }
    
    return predict_f1_race_simple('f1_race_dataset_initial.csv', qualifying_data)

# Uncomment to run interactive prediction:
# quick_predict()