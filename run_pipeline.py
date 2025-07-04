import os
from ai_predictor import train_model
import pandas as pd
import chardet
import numpy as np

def detect_encoding(file_path):
    """Detect the encoding of the file to handle different formats."""
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def generate_target_column(df):
    """Generate a target column dynamically if missing."""
    if 'target' not in df.columns:
        print("⚠️ 'target' column not found. Generating a dynamic target column...")
        # Use discountedSellingPrice * availableQuantity as a base, with some variability
        df['target'] = df['discountedSellingPrice'] * df['availableQuantity'] * (1 + np.random.uniform(-0.1, 0.1))
        print("✅ Dynamic target column created based on discountedSellingPrice * availableQuantity.")
    return df

def main():
    """Run the complete Quantum-AI optimization pipeline with dynamic target generation."""
    
    print("🚀 Starting Quantum-AI Supply Chain Optimizer Pipeline")
    print("=" * 60)
    
    # Define the data file path
    csv_path = "data/sample_data.csv"
    
    # Check if data file exists
    if not os.path.exists(csv_path):
        print("❌ Error: data/sample_data.csv not found!")
        print("Please create the data folder and CSV file first.")
        return
    
    # Attempt to read the file with detected encoding
    try:
        encoding = detect_encoding(csv_path)
        print(f"📊 Detected encoding: {encoding}")
        
        # Read the CSV with the detected encoding, fallback to latin-1 if utf-8 fails
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError:
            print("⚠️ UTF-8 failed, trying latin-1 encoding...")
            df = pd.read_csv(csv_path, encoding='latin-1')
        
        print(f"📈 Dataset shape: {df.shape}")
        print(f"🎯 Available columns: {list(df.columns)}")
        
        # Generate target column if missing
        df = generate_target_column(df)
        
        # Save the processed dataframe to a new CSV for verification
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv('data/processed/processed_data.csv', index=False)
        print("✅ Data processed and saved to data/processed/processed_data.csv")
        
        # Train the model with the processed data
        model, selected_features = train_model('data/processed/processed_data.csv')
        print("\n" + "=" * 60)
        print("✅ Pipeline completed successfully!")
        print("🌐 Now run: streamlit run app.py")
        print("=" * 60)
    
    except Exception as e:
        print(f"❌ Error during training: {e}")

if __name__ == "__main__":
    main()