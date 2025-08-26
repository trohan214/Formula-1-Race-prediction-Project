# F1 Race Prediction System - Usage Example
# How to use the prediction system with your data

from f1_prediction_system import F1RacePredictionSystem
import pandas as pd

def main():
    """Complete example of how to use the F1 prediction system"""
    
    print("🏎️  F1 RACE PREDICTION SYSTEM - USAGE EXAMPLE")
    print("="*55)
    
    # Initialize the prediction system
    predictor = F1RacePredictionSystem()
    
    # Load your CSV data
    print("📂 Loading data...")
    df = predictor.load_data('f1_race_dataset_initial.csv')  # Your CSV file
    
    # Create features for machine learning
    print("🔧 Creating features...")
    df_features = predictor.create_features(df)
    
    # Train the model
    print("🤖 Training models...")
    best_model = predictor.train_models(df_features)
    
    # Example 1: Predict based on new qualifying session
    print("\n" + "="*55)
    print("EXAMPLE 1: Predicting next race based on qualifying")
    print("="*55)
    
    # Create sample qualifying data (replace with real qualifying results)
    new_qualifying = pd.DataFrame({
        'Abbreviation': ['VER', 'HAM', 'LEC', 'SAI', 'RUS', 'NOR', 'PER', 'ALO', 'PIA', 'ALB'],
        'TeamName': ['Red Bull Racing', 'Mercedes', 'Ferrari', 'Ferrari', 'Mercedes', 
                    'McLaren', 'Red Bull Racing', 'Aston Martin', 'McLaren', 'Williams'],
        'QualGrid': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'QualTime_seconds': [78.5, 78.8, 79.1, 79.2, 79.0, 79.5, 78.9, 79.7, 79.8, 80.2]
    })
    
    # Get top 3 predictions
    results = predictor.get_top3_prediction(new_qualifying, "Belgian Grand Prix 2024")
    
    # Example 2: Get detailed predictions
    print("\n" + "="*55)
    print("EXAMPLE 2: Detailed prediction analysis")
    print("="*55)
    
    detailed_results = predictor.predict_race(new_qualifying, "Belgian Grand Prix 2024")
    
    print("Detailed Results DataFrame:")
    print(detailed_results[['Driver', 'Team', 'QualGrid', 'Top3_Probability', 'Win_Probability']].round(3))
    
    # Save predictions to CSV
    detailed_results.to_csv('race_predictions.csv', index=False)
    print(f"\n💾 Predictions saved to: race_predictions.csv")
    
    # Example 3: Historical analysis
    print("\n" + "="*55)
    print("EXAMPLE 3: Historical performance insights")
    print("="*55)
    
    predictor.analyze_historical_performance()
    
    # Example 4: Different qualifying scenarios
    print("\n" + "="*55)
    print("EXAMPLE 4: Testing different qualifying scenarios")
    print("="*55)
    
    # Scenario 1: Mercedes front row lockout
    scenario1 = pd.DataFrame({
        'Abbreviation': ['HAM', 'RUS', 'VER', 'LEC', 'SAI', 'NOR'],
        'TeamName': ['Mercedes', 'Mercedes', 'Red Bull Racing', 'Ferrari', 'Ferrari', 'McLaren'],
        'QualGrid': [1, 2, 3, 4, 5, 6],
        'QualTime_seconds': [78.1, 78.2, 78.5, 79.0, 79.1, 79.3]
    })
    
    print("Scenario 1: Mercedes 1-2 qualifying")
    predictor.get_top3_prediction(scenario1, "Scenario 1")
    
    # Scenario 2: Ferrari front row
    scenario2 = pd.DataFrame({
        'Abbreviation': ['LEC', 'SAI', 'VER', 'HAM', 'RUS', 'NOR'],
        'TeamName': ['Ferrari', 'Ferrari', 'Red Bull Racing', 'Mercedes', 'Mercedes', 'McLaren'],
        'QualGrid': [1, 2, 3, 4, 5, 6],
        'QualTime_seconds': [78.0, 78.1, 78.4, 78.8, 78.9, 79.2]
    })
    
    print("Scenario 2: Ferrari 1-2 qualifying")
    predictor.get_top3_prediction(scenario2, "Scenario 2")
    
    print("\n🎯 PREDICTION TIPS:")
    print("-" * 20)
    print("1. Qualifying position is the strongest predictor")
    print("2. Team performance (Red Bull > Mercedes > Ferrari) matters significantly")
    print("3. Driver experience and recent form influence predictions")
    print("4. The model is most confident about top-tier drivers from leading teams")
    print("5. Grid position P1-P3 have highest probability of podium finishes")
    
    print("\n✅ System ready for race predictions!")

def predict_custom_race():
    """Function to make predictions with custom data"""
    print("\n🔮 CUSTOM RACE PREDICTION")
    print("="*30)
    
    # You can modify this data based on real qualifying results
    custom_qualifying = pd.DataFrame({
        'Abbreviation': input("Enter driver abbreviations (comma-separated, e.g., VER,HAM,LEC): ").split(','),
        'TeamName': input("Enter team names (comma-separated): ").split(','),
        'QualGrid': list(map(int, input("Enter grid positions (comma-separated, e.g., 1,2,3): ").split(','))),
        'QualTime_seconds': list(map(float, input("Enter qualifying times in seconds (comma-separated): ").split(',')))
    })
    
    race_name = input("Enter race name: ")
    
    # Load and train the model
    predictor = F1RacePredictionSystem()
    df = predictor.load_data('f1_race_dataset_initial.csv')
    df_features = predictor.create_features(df)
    predictor.train_models(df_features)
    
    # Make prediction
    results = predictor.get_top3_prediction(custom_qualifying, race_name)
    
    return results

if __name__ == "__main__":
    # Run the main example
    main()
    
    # Uncomment the line below to enable interactive custom predictions
    # predict_custom_race()