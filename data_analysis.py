import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

FILE_PATH = 'All_Diets.csv'
MACRO_COLS = ['Protein(g)', 'Carbs(g)', 'Fat(g)']

def load_and_clean_data(file_path):
    """Loads the dataset and cleans missing/infinite values."""
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}. Please check the path.")
        return None

    print(f"Initial shape: {df.shape}")

    print("Cleaning data: Filling missing macronutrient values with the column mean...")
    for col in MACRO_COLS:
        df[col].fillna(df[col].mean(), inplace=True)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    print("Data loading and cleaning complete.")
    return df

def perform_core_analysis(df):
    """Performs all required data calculations and insights."""
    
    print("\n--- 1. Average Macronutrient Content per Diet Type ---")
    avg_macros = df.groupby('Diet_type')[MACRO_COLS].mean().round(2)
    print(avg_macros)

    top_protein = df.sort_values('Protein(g)', ascending=False).groupby('Diet_type').head(5)
    print("\n--- 2. Top 5 Protein-Rich Recipes per Diet Type ---")
    print(top_protein[['Diet_type', 'Recipe_name', 'Protein(g)']])

    highest_protein_recipe = df.loc[df['Protein(g)'].idxmax()]
    highest_protein_diet = highest_protein_recipe['Diet_type']
    print(f"\n--- 3. Diet Type with the Highest Single Protein Recipe ---")
    print(f"The diet type with the highest protein recipe is: {highest_protein_diet} ({highest_protein_recipe['Protein(g)']:.2f}g)")

    print("\n--- 4. Most Common Cuisine for Each Diet Type ---")
    most_common_cuisine = df.groupby('Diet_type')['Cuisine_type'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
    )
    print(most_common_cuisine.to_string())

    print("\n--- 5. Computing New Ratio Metrics ---")
    
    df['Carbs(g)_safe'] = df['Carbs(g)'].replace(0, 0.001)
    df['Fat(g)_safe'] = df['Fat(g)'].replace(0, 0.001)
    
    df['Protein_to_Carbs_ratio'] = (df['Protein(g)'] / df['Carbs(g)_safe']).round(3)
    df['Carbs_to_Fat_ratio'] = (df['Carbs(g)'] / df['Fat(g)_safe']).round(3)

    df.drop(columns=['Carbs(g)_safe', 'Fat(g)_safe'], inplace=True)
    
    print("\nFirst 5 rows showing calculated ratios:")
    print(df[['Recipe_name', 'Protein_to_Carbs_ratio', 'Carbs_to_Fat_ratio']].head())

    return avg_macros, top_protein, df

def visualize_results(avg_macros, top_protein, full_df):
    """Generates the required charts using Matplotlib and Seaborn and saves them as PNG files."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.figure(figsize=(14, 6))
    avg_macros.plot(kind='bar', ax=plt.gca(), rot=45)
    plt.title('Average Macronutrient Content by Diet Type')
    plt.ylabel('Average Amount (g)')
    plt.xlabel('Diet Type')
    plt.tight_layout()
    plt.savefig('bar_chart_avg_macros.png')
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_macros, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, cbar_kws={'label': 'Average Grams'})
    plt.title('Heatmap of Average Macronutrient Content by Diet Type')
    plt.ylabel('Diet Type')
    plt.xlabel('Macronutrient')
    plt.tight_layout()
    plt.savefig('heatmap_macros_diet.png')
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=top_protein, 
        x='Protein(g)', 
        y='Carbs(g)', 
        hue='Cuisine_type', 
        size='Fat(g)', 
        sizes=(50, 400),
        style='Diet_type',
        alpha=0.7
    )
    plt.title('Top 5 Protein Recipes: Distribution by Macros, Cuisine, and Diet')
    plt.xlabel('Protein (g)')
    plt.ylabel('Carbs (g)')
    plt.legend(title='Cuisine & Diet', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('scatter_plot_top_protein.png')
    plt.close()
    
    print("\nVisualizations saved: bar_chart_avg_macros.png, heatmap_macros_diet.png, scatter_plot_top_protein.png")


if __name__ == '__main__':
    data_frame = load_and_clean_data(FILE_PATH)
    
    if data_frame is not None:
        avg_macros_df, top_protein_df, full_data_df = perform_core_analysis(data_frame.copy())
        
        visualize_results(avg_macros_df, top_protein_df, full_data_df)
        
        print("\nTask 1 Analysis and Visualization complete. Check your directory for PNG files.")