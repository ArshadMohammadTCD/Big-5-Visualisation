import pandas as pd
import numpy as np

def transform_columns_to_percentiles(input_file, output_file, columns_to_transform=None):
    """
    Transform specified columns in a CSV file to percentile values.
    
    Parameters:
    - input_file (str): Path to the input CSV file
    - output_file (str): Path to save the transformed CSV file
    - columns_to_transform (list, optional): List of column names to transform. 
      If None, transforms all numeric columns between 0 and 1.
    
    Returns:
    - pandas.DataFrame: Transformed DataFrame
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # If no columns specified, find suitable columns
    if columns_to_transform is None:
        columns_to_transform = [
            col for col in df.columns 
            if df[col].dtype in ['float64', 'float32'] 
            and ((df[col] >= 0) & (df[col] <= 1)).all()
        ]
    
    # Validate specified columns
    for col in columns_to_transform:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the CSV file")
        
        # Check if column values are between 0 and 1
        if not np.all((df[col] >= 0) & (df[col] <= 1)):
            raise ValueError(f"Column '{col}' must contain values between 0 and 1")
    
    # Transform specified columns to percentiles
    for col in columns_to_transform:
        df[col] = (df[col].rank(method='average', pct=True) * 100).round(2)
    
    # Save the transformed DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    
    return df

# Example usage
if __name__ == "__main__":
    try:
        # Specific columns for personality scores
        personality_columns = [
            'agreeable_score', 
            'extraversion_score', 
            'openness_score', 
            'conscientiousness_score', 
            'neuroticism_score'
        ]
        
        # Replace these with your actual file paths
        input_file = 'big_five_scores.csv'
        output_file = 'transformed_personality_scores.csv'
        
        # Transform the specified columns
        transformed_df = transform_columns_to_percentiles(
            input_file, 
            output_file, 
            columns_to_transform=personality_columns
        )
        
        print("Columns transformed to percentiles:")
        print(", ".join(personality_columns))
        print("\nTransformed data preview:")
        print(transformed_df.head())
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Optional additional utility function
def preview_percentile_transformation(df, columns):
    """
    Create a side-by-side preview of original and percentile-transformed values
    
    Parameters:
    - df (pandas.DataFrame): Original DataFrame
    - columns (list): Columns to preview
    
    Returns:
    - pandas.DataFrame: Preview of original and transformed values
    """
    preview = df.copy()
    
    for col in columns:
        percentile_col = f'{col}_percentile'
        preview[percentile_col] = df[col].rank(method='average', pct=True)
    
    return preview[[col for pair in zip(columns, [f'{col}_percentile' for col in columns]) for col in pair]]