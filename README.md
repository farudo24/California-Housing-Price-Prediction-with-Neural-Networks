# California-Housing-Price-Prediction-with-Neural-Networks
This project implements a deep neural network regression model to predict California housing prices based on the 1990 census data
"""
California Housing Price Prediction with Neural Networks - Google Colab Version
Author: [Your Name]
Date: December 2024

This script implements a deep learning regression model to predict
California housing prices based on the 1990 census data.
Based on concepts from "Understanding Deep Learning" (Chapters 1-9)
"""

# ============================================================================
# 0. INSTALL AND IMPORT NECESSARY PACKAGES
# ============================================================================

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

print("="*60)
print("PACKAGES IMPORTED SUCCESSFULLY")
print("="*60)
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

def load_data():
    """
    Load California Housing dataset.
    """
    print("="*60)
    print("LOADING DATASET")
    print("="*60)

    # Since you already uploaded housing.csv, we'll load it directly
    try:
        df = pd.read_csv('housing.csv')
        print("Dataset loaded successfully from uploaded file!")
    except:
        # If not available, use sklearn dataset
        try:
            from sklearn.datasets import fetch_california_housing
            california = fetch_california_housing()
            df = pd.DataFrame(california.data, columns=california.feature_names)
            df['median_house_value'] = california.target * 100000
            print("Dataset loaded from sklearn successfully!")
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return None

    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# ============================================================================
# 2. DATA EXPLORATION FUNCTION
# ============================================================================

def explore_data(df):
    """
    Perform exploratory data analysis.
    """
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)

    # Display basic information
    print("\nDataset Information:")
    print(df.info())

    print("\nFirst 5 rows:")
    display(df.head())

    print("\nStatistical Summary:")
    display(df.describe())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Check categorical column
    if 'ocean_proximity' in df.columns:
        print("\nOcean Proximity Categories:")
        print(df['ocean_proximity'].value_counts())

    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Distribution of target variable
    axes[0, 0].hist(df['median_house_value'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Median House Value ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of House Prices')
    axes[0, 0].grid(True, alpha=0.3)

    # Correlation heatmap for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        corr_matrix = df[numerical_cols].corr()
        # Use seaborn for heatmap
        import seaborn as sns
        sns.heatmap(corr_matrix, ax=axes[0, 1], annot=True, fmt='.2f', cmap='coolwarm')
        axes[0, 1].set_title('Correlation Heatmap (Numerical Features)')

    # Price vs Median Income
    axes[0, 2].scatter(df['median_income'], df['median_house_value'], alpha=0.3)
    axes[0, 2].set_xlabel('Median Income')
    axes[0, 2].set_ylabel('House Price ($)')
    axes[0, 2].set_title('Price vs Median Income')
    axes[0, 2].grid(True, alpha=0.3)

    # Price vs House Age
    axes[1, 0].scatter(df['housing_median_age'], df['median_house_value'], alpha=0.3)
    axes[1, 0].set_xlabel('House Age (years)')
    axes[1, 0].set_ylabel('House Price ($)')
    axes[1, 0].set_title('Price vs House Age')
    axes[1, 0].grid(True, alpha=0.3)

    # Price vs Location
    if 'latitude' in df.columns and 'longitude' in df.columns:
        scatter = axes[1, 1].scatter(df['longitude'], df['latitude'],
                                     c=df['median_house_value'],
                                     cmap='jet', alpha=0.5, s=10)
        axes[1, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Latitude')
        axes[1, 1].set_title('Geographic Distribution of Prices')
        plt.colorbar(scatter, ax=axes[1, 1], label='Price ($)')

    # Boxplot for ocean proximity
    if 'ocean_proximity' in df.columns:
        import seaborn as sns
        sns.boxplot(x='ocean_proximity', y='median_house_value', data=df, ax=axes[1, 2])
        axes[1, 2].set_title('Price by Ocean Proximity')
        axes[1, 2].set_ylabel('Price ($)')
        axes[1, 2].set_xlabel('Ocean Proximity')
        axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    return df

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """
    Preprocess the housing dataset for neural network.
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)

    # Create copy
    data = df.copy()

    # 1. Handle missing values
    print("1. Handling missing values...")
    if data.isnull().sum().any():
        missing_cols = data.columns[data.isnull().any()].tolist()
        print(f"   Columns with missing values: {missing_cols}")

        # Use median for numerical columns
        for col in missing_cols:
            if data[col].dtype in ['int64', 'float64']:
                data[col].fillna(data[col].median(), inplace=True)
                print(f"   Filled {col} with median: {data[col].median():.2f}")

    # 2. Separate features and target
    print("\n2. Separating features and target...")
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']

    # 3. Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"   Numerical features: {numerical_cols}")
    print(f"   Categorical features: {categorical_cols}")

    # 4. Create preprocessing pipeline
    print("\n3. Creating preprocessing pipeline...")

    # Numerical preprocessing: scaling
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Categorical preprocessing: one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # 5. Feature engineering (add new features)
    print("4. Performing feature engineering...")

    # Create new features before splitting
    if all(col in X.columns for col in ['total_rooms', 'households']):
        X['rooms_per_household'] = X['total_rooms'] / X['households']
        numerical_cols.append('rooms_per_household')
        print("   Created: rooms_per_household")

    if all(col in X.columns for col in ['total_bedrooms', 'total_rooms']):
        X['bedrooms_per_room'] = X['total_bedrooms'] / X['total_rooms']
        numerical_cols.append('bedrooms_per_room')
        print("   Created: bedrooms_per_room")

    if all(col in X.columns for col in ['population', 'households']):
        X['population_per_household'] = X['population'] / X['households']
        numerical_cols.append('population_per_household')
        print("   Created: population_per_household")

    # Update the preprocessor with new numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # 6. Split data
    print("\n5. Splitting data into train/validation/test sets...")

    # First split: 80% train+val, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Second split: 75% train, 25% validation (of the 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )

    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    print(f"   Test set: {X_test.shape}")

    # 7. Fit preprocessor on training data and transform all sets
    print("\n6. Fitting preprocessor and transforming data...")

    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    # For numerical features
    num_feature_names = numerical_cols

    # For categorical features (after one-hot encoding)
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
        all_feature_names = list(num_feature_names) + list(cat_feature_names)
    else:
        all_feature_names = num_feature_names

    print(f"   Number of features after preprocessing: {len(all_feature_names)}")

    # 8. Scale target variable
    print("7. Scaling target variable...")
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

    print("\nPreprocessing complete!")

    return (X_train_processed, X_val_processed, X_test_processed,
            y_train_scaled, y_val_scaled, y_test_scaled,
            target_scaler, all_feature_names, preprocessor)

# ============================================================================
# 4. NEURAL NETWORK MODEL BUILDING
# ============================================================================

def build_model(input_dim,
                hidden_layers=[128, 64, 32],
                activation='relu',
                dropout_rate=0.2,
                l2_reg=0.001,
                learning_rate=0.001,
                use_batch_norm=True):
    """
    Build and compile a neural network model.
    """
    print("\n" + "="*60)
    print("BUILDING NEURAL NETWORK MODEL")
    print("="*60)
    print(f"Input dimension: {input_dim}")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Activation: {activation}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"L2 regularization: {l2_reg}")
    print(f"Learning rate: {learning_rate}")

    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=(input_dim,)))

    # Hidden layers
    for i, units in enumerate(hidden_layers):
        # Dense layer with L2 regularization
        model.add(layers.Dense(
            units=units,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'dense_{i+1}'
        ))

        # Batch normalization
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))

        # Dropout
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))

    # Output layer (single neuron for regression)
    model.add(layers.Dense(1, activation='linear', name='output'))

    # Compile model
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999
    )

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )

    print("\nModel Summary:")
    model.summary()

    return model

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=200, batch_size=32, patience=20):
    """
    Train the model with callbacks.
    """
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)

    # Callbacks
    callbacks_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-6,
            verbose=1
        ),

        # Model checkpoint
        callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )

    return history, model

# ============================================================================
# 6. VISUALIZE TRAINING HISTORY
# ============================================================================

def plot_training_history(history):
    """
    Plot training history.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Loss Over Epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MAE
    axes[0, 1].plot(history.history['mae'], label='Training MAE')
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Mean Absolute Error Over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)

    # Final epoch markers
    best_epoch = np.argmin(history.history['val_loss'])
    axes[0, 0].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # Print best results
    print(f"\nBest epoch: {best_epoch + 1}")
    print(f"Best training loss: {history.history['loss'][best_epoch]:.4f}")
    print(f"Best validation loss: {history.history['val_loss'][best_epoch]:.4f}")
    print(f"Best validation MAE: {history.history['val_mae'][best_epoch]:.4f}")

# ============================================================================
# 7. MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, target_scaler):
    """
    Evaluate model performance.
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)

    # Predict
    y_pred_scaled = model.predict(X_test)

    # Inverse transform
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_actual = target_scaler.inverse_transform(y_test)

    # Calculate metrics
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    # Percentage errors
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred - y_actual) /
                         (np.abs(y_actual) + np.abs(y_pred)))

    print(f"Mean Squared Error (MSE): ${mse:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Symmetric MAPE (SMAPE): {smape:.2f}%")

    # Create evaluation plots
    create_evaluation_plots(y_actual, y_pred, mae, rmse, r2)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'smape': smape
    }, y_pred, y_actual

def create_evaluation_plots(y_actual, y_pred, mae, rmse, r2):
    """
    Create evaluation visualizations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_actual, y_pred, alpha=0.5, s=10)
    axes[0, 0].plot([y_actual.min(), y_actual.max()],
                    [y_actual.min(), y_actual.max()],
                    'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Price ($)')
    axes[0, 0].set_ylabel('Predicted Price ($)')
    axes[0, 0].set_title(f'Actual vs Predicted\nMAE: ${mae:,.0f}, R²: {r2:.3f}')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residuals
    residuals = y_actual - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Price ($)')
    axes[0, 1].set_ylabel('Residuals ($)')
    axes[0, 1].set_title(f'Residual Plot\nRMSE: ${rmse:,.0f}')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Error distribution
    axes[1, 0].hist(residuals.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Prediction Error ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Prediction Errors')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Percentage error distribution
    percent_errors = (residuals / y_actual) * 100
    axes[1, 1].hist(percent_errors.flatten(), bins=50,
                    edgecolor='black', alpha=0.7, range=(-100, 100))
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Percentage Error (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Percentage Errors')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ============================================================================
# 8. SAMPLE PREDICTIONS
# ============================================================================

def show_sample_predictions(y_actual, y_pred, n_samples=10):
    """
    Show sample predictions.
    """
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)

    # Random samples
    indices = np.random.choice(len(y_actual), size=n_samples, replace=False)

    results = []
    for idx in indices:
        actual = y_actual[idx][0]
        predicted = y_pred[idx][0]
        error = actual - predicted
        error_pct = (error / actual) * 100

        results.append({
            'Index': idx,
            'Actual ($)': f"${actual:,.0f}",
            'Predicted ($)': f"${predicted:,.0f}",
            'Error ($)': f"${error:,.0f}",
            'Error (%)': f"{error_pct:.1f}%"
        })

    results_df = pd.DataFrame(results)
    display(results_df)

# ============================================================================
# 9. SAVE MODEL AND RESULTS
# ============================================================================

def save_results(model, target_scaler, metrics):
    """
    Save model and results.
    """
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save model
    model.save('california_housing_model.h5')
    print("✓ Model saved as 'california_housing_model.h5'")

    # Save scalers
    import joblib
    joblib.dump(target_scaler, 'target_scaler.pkl')
    print("✓ Target scaler saved as 'target_scaler.pkl'")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('model_metrics.csv', index=False)
    print("✓ Metrics saved as 'model_metrics.csv'")

    # Download files to local machine
    from google.colab import files

    print("\nTo download files, run:")
    print("files.download('california_housing_model.h5')")
    print("files.download('target_scaler.pkl')")
    print("files.download('model_metrics.csv')")

# ============================================================================
# 10. MAIN PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline.
    """
    print("="*60)
    print("CALIFORNIA HOUSING PRICE PREDICTION - GOOGLE COLAB")
    print("="*60)

    # Step 1: Load data
    df = load_data()
    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Step 2: Explore data
    df = explore_data(df)

    # Step 3: Preprocess data
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     target_scaler, feature_names, preprocessor) = preprocess_data(df)

    print(f"\nFinal data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"y_test: {y_test.shape}")

    # Step 4: Build model
    model = build_model(
        input_dim=X_train.shape[1],
        hidden_layers=[128, 64, 32],
        activation='relu',
        dropout_rate=0.2,
        l2_reg=0.001,
        learning_rate=0.001,
        use_batch_norm=True
    )

    # Step 5: Train model
    history, model = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=200,
        batch_size=32,
        patience=20
    )

    # Step 6: Plot training history
    plot_training_history(history)

    # Step 7: Evaluate model
    metrics, y_pred, y_actual = evaluate_model(
        model, X_test, y_test, target_scaler
    )

    # Step 8: Show sample predictions
    show_sample_predictions(y_actual, y_pred, n_samples=10)

    # Step 9: Save results
    save_results(model, target_scaler, metrics)

    # Summary
    print("\n" + "="*60)
    print("DEEP LEARNING CONCEPTS APPLIED")
    print("="*60)
    print("""
    From "Understanding Deep Learning" (Chapters 1-9):

    1. Supervised Learning Framework (Chapter 1-2)
       - Regression problem setup
       - Training/validation/test split

    2. Neural Network Architecture (Chapter 3-4)
       - Deep feedforward network
       - Multiple hidden layers with ReLU activation

    3. Loss Functions (Chapter 5)
       - Mean Squared Error (MSE) for regression
       - Proper scaling of inputs and outputs

    4. Optimization (Chapter 6)
       - Adam optimizer with momentum
       - Learning rate scheduling
       - Batch processing

    5. Backpropagation (Chapter 7)
       - Automatic gradient computation (handled by TensorFlow)

    6. Model Evaluation (Chapter 8)
       - Multiple evaluation metrics (MSE, MAE, R², MAPE)
       - Residual analysis

    7. Regularization (Chapter 9)
       - L2 regularization (weight decay)
       - Dropout layers
       - Early stopping
       - Batch normalization
    """)

    print("\n" + "="*60)
    print("PROJECT COMPLETE!")
    print("="*60)

# ============================================================================
# RUN THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    main()
