import pandas as pd
import numpy as np

def aggregate_personality_scores_by_age(input_file, output_file, min_entries=30, personality_columns=None):
    """
    Aggregate transformed personality scores by age, filtering out countries with few entries.
    
    Parameters:
    - input_file (str): Path to the input CSV file with transformed personality scores
    - output_file (str): Path to save the aggregated age personality scores
    - min_entries (int): Minimum number of entries required for a age to be included
    - personality_columns (list, optional): List of personality score columns to aggregate
    
    Returns:
    - pandas.DataFrame: Aggregated DataFrame with average personality scores by age
    """
    # Default personality columns if not specified
    if personality_columns is None:
        personality_columns = [
            'agreeable_score', 
            'extraversion_score', 
            'openness_score', 
            'conscientiousness_score', 
            'neuroticism_score'
        ]
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Validate columns
    required_columns = personality_columns + ['age']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the CSV file")
    
    # Count entries per age and filter
    age_entry_counts = df['age'].value_counts()
    countries_with_enough_entries = age_entry_counts[age_entry_counts >= min_entries].index
    
    # Filter DataFrame to include only countries with enough entries
    filtered_df = df[df['age'].isin(countries_with_enough_entries)]
    
    # Group by age and calculate mean for each personality score
    age_personality_avg = filtered_df.groupby('age')[personality_columns].mean().reset_index()
    
    # Round to 2 decimal places
    for col in personality_columns:
        age_personality_avg[col] = age_personality_avg[col].round(2)
    
    # Add a column for the number of entries per age
    age_personality_avg['entries_count'] = age_entry_counts[age_personality_avg['age']].values
    
    # Save to CSV
    age_personality_avg.to_csv(output_file, index=False)
    
    return age_personality_avg

# Example usage
if __name__ == "__main__":
    try:
        # Personality columns to aggregate
        personality_columns = [
            'agreeable_score', 
            'extraversion_score', 
            'openness_score', 
            'conscientiousness_score', 
            'neuroticism_score'
        ]
        
        # Replace these with your actual file paths
        input_file = 'transformed_personality_scores.csv'
        output_file = 'age_personality_averages.csv'
        
        # Aggregate personality scores by age, only including countries with 30+ entries
        age_averages = aggregate_personality_scores_by_age(
            input_file, 
            output_file, 
            min_entries=100,
            personality_columns=personality_columns
        )
        
        print("age Personality Score Averages (30+ entries):")
        print(age_averages)
        
        print(f"\nAggregated data saved to {output_file}")
        print(f"Total countries included: {len(age_averages)}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Utility function to preview age filtering
def preview_age_entry_counts(df):
    """
    Show the number of entries for each age
    
    Parameters:
    - df (pandas.DataFrame): Original DataFrame
    
    Returns:
    - pandas.Series: Count of entries per age
    """
    return df['age'].value_counts()