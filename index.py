# index.py
# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and perform initial data cleaning"""
        self.df = pd.read_csv(self.data_path)
        
        # Drop any NA values
        self.df.dropna(inplace=True)
        
        # Print dataset info
        print('Summary of dataset:\n')
        print(self.df.info())
        
        # Display sample data
        print('\nSample of dataset:\n')
        print(self.df.sample(frac=1).head())
        
        return self.df
    
    def prepare_features(self):
        """Prepare features for model training"""
        # Extract features (text) and labels
        if 'text' in self.df.columns and 'label' in self.df.columns:
            X = self.df['text'].tolist()
            y = self.df['label'].tolist()
        else:
            raise ValueError("Required columns 'text' and 'label' not found in dataset")
            
        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def create_pipeline_components(self):
        """Create text processing pipeline components"""
        # Initialize the vectorizer and transformer
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        
        return vectorizer, transformer

# Example usage:
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor('FakeNews.csv')
    
    # Load and clean data
    df = preprocessor.load_data()
    
    # Prepare features
    X_train, X_test, y_train, y_test = preprocessor.prepare_features()
    
    # Get pipeline components
    vectorizer, transformer = preprocessor.create_pipeline_components()
