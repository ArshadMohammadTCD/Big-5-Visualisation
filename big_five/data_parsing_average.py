import pandas as pd
import numpy as np

def aggregate_personality_scores_by_country(input_file, output_file, min_entries=30, personality_columns=None):
    """
    Aggregate transformed personality scores by country, filtering out countries with few entries.
    
    Parameters:
    - input_file (str): Path to the input CSV file with transformed personality scores
    - output_file (str): Path to save the aggregated country personality scores
    - min_entries (int): Minimum number of entries required for a country to be included
    - personality_columns (list, optional): List of personality score columns to aggregate
    
    Returns:
    - pandas.DataFrame: Aggregated DataFrame with average personality scores by country
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
    required_columns = personality_columns + ['country']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the CSV file")
    
    # Count entries per country and filter
    country_entry_counts = df['country'].value_counts()
    countries_with_enough_entries = country_entry_counts[country_entry_counts >= min_entries].index
    
    # Filter DataFrame to include only countries with enough entries
    filtered_df = df[df['country'].isin(countries_with_enough_entries)]
    
    # Group by country and calculate mean for each personality score
    country_personality_avg = filtered_df.groupby('country')[personality_columns].mean().reset_index()
    
    # Round to 2 decimal places
    for col in personality_columns:
        country_personality_avg[col] = country_personality_avg[col].round(2)
    
    # Add a column for the number of entries per country
    country_personality_avg['entries_count'] = country_entry_counts[country_personality_avg['country']].values
    
    # Save to CSV
    country_personality_avg.to_csv(output_file, index=False)
    
    return country_personality_avg

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
        output_file = 'country_personality_averages.csv'
        
        # Aggregate personality scores by country, only including countries with 30+ entries
        country_averages = aggregate_personality_scores_by_country(
            input_file, 
            output_file, 
            min_entries=100,
            personality_columns=personality_columns
        )
        
        print("Country Personality Score Averages (30+ entries):")
        print(country_averages)
        
        print(f"\nAggregated data saved to {output_file}")
        print(f"Total countries included: {len(country_averages)}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Utility function to preview country filtering
def preview_country_entry_counts(df):
    """
    Show the number of entries for each country
    
    Parameters:
    - df (pandas.DataFrame): Original DataFrame
    
    Returns:
    - pandas.Series: Count of entries per country
    """
    return df['country'].value_counts()