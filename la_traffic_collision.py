"""
PROJECT: Predictive Analysis of Traffic Collision Risks in Los Angeles
AUTHOR: Marcel G
DESCRIPTION:
    This project utilizes Machine Learning techniques to analyze driver behavioral patterns.
    The goal is to build a classification model to predict the time of occurrence (Day/Night)
    based on demographic and geographic features.
    The analysis highlights a significant correlation between driver age and nighttime driving risk.

TECH STACK: Python, Pandas, Scikit-Learn (DecisionTree, RandomForest, MLP), Seaborn.
DATA SOURCE: Los Angeles Open Data (Traffic Collision Data from 2010 to Present).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configuration
DATA_URL = "https://data.lacity.org/api/views/d5tf-ez2w/rows.csv?accessType=DOWNLOAD"
SELECTED_COLUMNS = ['Date Occurred', 'Time Occurred', 'Area Name', 'Victim Age', 'Victim Sex', 'Location']
RANDOM_SEED = 42

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_clean_data(url: str) -> pd.DataFrame:
    """
    Loads data from the API and performs initial cleaning and filtering.
    """
    print(f"[INFO] Loading data from: {url}...")
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return pd.DataFrame()

    # Filter columns
    df = df[[c for c in SELECTED_COLUMNS if c in df.columns]].copy()
    
    # 1. Filter Invalid Ages (Assuming 0-99 is the valid range for drivers/victims)
    df = df[(df['Victim Age'] > 0) & (df['Victim Age'] < 99)]
    
    # 2. Filter Sex
    df = df[df['Victim Sex'].isin(['M', 'F'])]

    print(f"[INFO] Data loaded successfully. Rows after initial cleaning: {len(df)}")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw columns into features suitable for ML models.
    """
    print("[INFO] Starting feature engineering...")
    
    # 1. Parse Hour
    # Pad string to 4 digits (e.g., '130' -> '0130') and take first 2 chars
    df['Hour'] = pd.to_numeric(
        df['Time Occurred'].astype(str).str.zfill(4).str[:2], 
        errors='coerce'
    )
    df = df.dropna(subset=['Hour'])
    
    # 2. Create Target Variable: Day vs Night (Cutoff at 18:00/6:00 PM)
    # Day: 06:00 - 17:59, Night: 18:00 - 05:59
    df['Time_Category'] = df['Hour'].apply(lambda h: 'Day' if 6 <= h < 18 else 'Night')
    
    # 3. Parse Coordinates
    # Cleaning location strings "(34.1, -118.2)" -> 34.1, -118.2
    # Using regex extraction for robustness
    coords = df['Location'].astype(str).str.extract(r'\((?P<lat>[-+]?\d*\.\d+),\s*(?P<lon>[-+]?\d*\.\d+)\)')
    df['Latitude'] = pd.to_numeric(coords['lat'])
    df['Longitude'] = pd.to_numeric(coords['lon'])
    
    # Filter invalid coordinates (0.0 often indicates missing data in this dataset)
    df = df[(df['Latitude'] != 0) & (df['Latitude'].notna())]

    # 4. Date Features
    df['Date'] = pd.to_datetime(df['Date Occurred'])
    df['Month'] = df['Date'].dt.month
    df['Is_Weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)

    # 5. Encoding
    le_sex = LabelEncoder()
    df['Sex_Code'] = le_sex.fit_transform(df['Victim Sex']) # 0/1
    
    le_target = LabelEncoder()
    df['Target_Code'] = le_target.fit_transform(df['Time_Category']) # 0=Day, 1=Night (usually, depends on alphabetical order)

    print(f"[INFO] Feature engineering complete. Final dataset shape: {df.shape}")
    return df

def train_models(df: pd.DataFrame):
    """
    Trains and compares Decision Tree, Random Forest, and MLP Classifier.
    """
    print("\n[INFO] Training models...")
    
    features = ['Latitude', 'Longitude', 'Victim Age', 'Sex_Code', 'Month', 'Is_Weekend']
    X = df[features]
    y = df['Target_Code']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # Scaling (Crucial for Neural Networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define Models
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_SEED),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_SEED),
        'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=RANDOM_SEED)
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        # Use scaled data for MLP, raw for Trees (though trees handle unscaled fine, keeping consistent flow)
        if name == 'MLP Neural Net':
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        predictions[name] = preds
        print(f"   -> {name} Accuracy: {acc:.2%}")

    return results, predictions, y_test

def create_visualizations(df: pd.DataFrame, results: dict, predictions: dict, y_test):
    """
    Generates insightful plots for analysis.
    """
    print("\n[INFO] Generating visualizations...")

    # 1. Geospatial Distribution
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, x='Longitude', y='Latitude', hue='Time_Category',
        palette={'Day': 'orange', 'Night': 'navy'}, alpha=0.2, s=15
    )
    plt.title('1. Accident Hotspots: Day vs. Night', fontsize=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Time of Day')
    plt.axis('equal')
    plt.show()

    # 2. Correlation Matrix
    plt.figure(figsize=(8, 6))
    corr_cols = ['Victim Age', 'Latitude', 'Longitude', 'Target_Code', 'Is_Weekend']
    
    # Rename columns for cleaner heatmap
    corr_df = df[corr_cols].rename(columns={'Target_Code': 'Is_Night', 'Victim Age': 'Driver Age'})
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-0.2, vmax=0.2)
    plt.title('2. Feature Correlation Matrix', fontsize=14)
    plt.show()

    # 3. Age Distribution by Time
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df[df['Time_Category']=='Day'], x='Victim Age', fill=True, label='Day', color='orange', alpha=0.5)
    sns.kdeplot(data=df[df['Time_Category']=='Night'], x='Victim Age', fill=True, label='Night', color='navy', alpha=0.5)
    plt.title('3. Driver Age Profile: Who drives when?', fontsize=14)
    plt.xlabel('Driver Age')
    plt.xlim(15, 90) # Focus on driving age
    plt.legend()
    plt.show()

    # 4. Gender Distribution
    props = df.groupby("Time_Category")['Victim Sex'].value_counts(normalize=True).unstack()
    props.plot(kind='bar', stacked=True, color=['pink', 'steelblue'], figsize=(8, 6))
    plt.title('4. Gender Distribution by Time of Day', fontsize=14)
    plt.ylabel('Proportion')
    plt.xlabel('Time Category')
    plt.legend(title='Sex', labels=['Female', 'Male'], bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

    # 5. Model Benchmark
    plt.figure(figsize=(8, 5))
    names = list(results.keys())
    values = list(results.values())
    bars = plt.bar(names, values, color=['skyblue', 'mediumseagreen', 'rebeccapurple'])
    plt.ylim(0.5, 0.9)
    plt.title('5. Model Performance Benchmark (Accuracy)', fontsize=14)
    
    # Add labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1%}', ha='center', va='bottom', weight='bold')
    plt.show()

    # 6. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['Blues', 'Greens', 'Purples']
    
    model_names = list(predictions.keys())
    
    for i, name in enumerate(model_names):
        cm = confusion_matrix(y_test, predictions[name])
        sns.heatmap(cm, annot=True, fmt='d', cmap=colors[i], ax=axes[i])
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual' if i == 0 else '')
        
    plt.suptitle('6. Model Error Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    print("=== LA TRAFFIC DRIVER ANALYSIS PROJECT ===")
    
    # Pipeline execution
    df_raw = load_and_clean_data(DATA_URL)
    
    if not df_raw.empty:
        df_processed = feature_engineering(df_raw)
        model_results, preds, y_test = train_models(df_processed)
        create_visualizations(df_processed, model_results, preds, y_test)
        
        best_model = max(model_results, key=model_results.get)
        print(f"\n[CONCLUSION] The best performing model is: {best_model} with {model_results[best_model]:.2%} accuracy.")
        print("Analysis Complete.")
    else:
        print("[ERROR] Analysis aborted due to data loading failure.")

if __name__ == "__main__":
    main()