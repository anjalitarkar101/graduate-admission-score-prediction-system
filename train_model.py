# ==========================================================
# train_model.py - Graduate Admission Prediction using ANN
# ==========================================================

### Data manipulation and numerical operations
import pandas as pd                  # For loading and manipulating CSV data
import numpy as np                   # For numerical operations and array handling

### Data visualization
import matplotlib.pyplot as plt      # For creating plots and charts
import seaborn as sns                # For statistical visualizations

### Machine learning utilities
from sklearn.model_selection import train_test_split      # For splitting data into train/test
from sklearn.preprocessing import MinMaxScaler            # For scaling features to 0-1 range
from sklearn.metrics import mean_squared_error, r2_score  # For model evaluation

### Deep learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

### System utilities
import os                   # For file and directory operations
import pickle               # For saving Python objects (scaler, feature names)
import warnings             # For managing warning messages


# Suppress warning messages to keep output clean
warnings.filterwarnings('ignore')

print("=" * 60)
print("🎓 GRADUATE ADMISSION PREDICTION using ANN")
print("=" * 60)

# ==========================================================
# 1. LOAD AND EXPLORE THE DATA
# ==========================================================
print("\n📥 Loading dataset...")

# Check if dataset exists
if not os.path.exists('data/Admission_Predict_Ver1.1.csv'):
    print("❌ Dataset not found! Please place 'Admission_Predict_Ver1.1.csv' in 'data/' folder.")
    exit()

# Load the dataset into a pandas DataFrame
df = pd.read_csv('data/Admission_Predict_Ver1.1.csv')

# Display dataset information
print(f"\n📊 Dataset Information:")
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print(f"\n   First 5 rows:")
print(df.head())

# Check for missing values
print(f"\n🔍 Missing values:\n{df.isnull().sum()}")

# Check for duplicates
print(f"\n🔍 Duplicate rows: {df.duplicated().sum()}")

# ==========================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================
print("\n📊 Exploratory Data Analysis...")

# Create a 2x3 grid of plots (6 plots total)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# ----- Plot 1: Distribution of Chance of Admit -----
axes[0, 0].hist(df['Chance of Admit'], bins=20, edgecolor='black', color='skyblue')
axes[0, 0].set_title('Distribution of Chance of Admit')
axes[0, 0].set_xlabel('Chance of Admit')
axes[0, 0].set_ylabel('Frequency')

# ----- Plot 2: GRE Score vs Chance of Admit -----
axes[0, 1].scatter(df['GRE Score'], df['Chance of Admit'], alpha=0.5, color='green')
axes[0, 1].set_title('GRE Score vs Chance of Admit')
axes[0, 1].set_xlabel('GRE Score')
axes[0, 1].set_ylabel('Chance of Admit')

# ----- Plot 3: CGPA vs Chance of Admit -----
axes[0, 2].scatter(df['CGPA'], df['Chance of Admit'], alpha=0.5, color='red')
axes[0, 2].set_title('CGPA vs Chance of Admit')
axes[0, 2].set_xlabel('CGPA')
axes[0, 2].set_ylabel('Chance of Admit')

# ----- Plot 4: Research vs Chance of Admit -----
research_means = df.groupby('Research')['Chance of Admit'].mean()
axes[1, 0].bar(['No Research', 'With Research'], research_means.values, color=['orange', 'blue'])
axes[1, 0].set_title('Research vs Chance of Admit')
axes[1, 0].set_ylabel('Average Chance of Admit')

# ----- Plot 5: TOEFL Score vs Chance of Admit -----
axes[1, 2].scatter(df['TOEFL Score'], df['Chance of Admit'], alpha=0.5, color='purple')
axes[1, 2].set_title('TOEFL Score vs Chance of Admit')
axes[1, 2].set_xlabel('TOEFL Score')
axes[1, 2].set_ylabel('Chance of Admit')

# ----- Plot 6: Correlation Heatmap -----
# Shows how all features are correlated with each other
# Values close to 1 = strong positive correlation
# Values close to -1 = strong negative correlation
corr = df.drop(columns=['Serial No.']).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Correlation Heatmap')

plt.tight_layout()             # Adjust layout to prevent overlapping labels
plt.savefig('saved_visualizations/eda_admission.png')
plt.show(block=False)          # Show without blocking execution
plt.pause(2)
plt.close()
print("✅ EDA plots saved as 'saved_visualizations/eda_admission.png'")

# ==========================================================
# 3. DATA PREPROCESSING
# ==========================================================
print("\n🔄 Preprocessing data...")

# Create a copy of the original DataFrame to avoid modifying it
df_processed = df.copy()
df_processed = df_processed.drop(columns=['Serial No.'], errors='ignore')    # errors='ignore' prevents crashing if column doesn't exist

print(f"   Features after dropping Serial No.: {df_processed.columns.tolist()}")

# Separate features (input)  and target (output)
X = df_processed.drop(columns=['Chance of Admit'])
Y = df_processed['Chance of Admit'].values       # .values converts to numpy array for TensorFlow


print(f"   Features shape: {X.shape}")
print(f"   Target shape: {Y.shape}")

# ==========================================================
# 4. TRAIN-TEST SPLIT
# ==========================================================
print("\n🔄 Splitting data into train/test sets...")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# ==========================================================
# 5. FEATURE SCALING (MinMaxScaler for regression)
# ==========================================================
print("\n🔄 Scaling features...")
scaler = MinMaxScaler()

# fit_transform on training data: learns the scale AND applies it
X_train_scaled = scaler.fit_transform(X_train)

# transform on test data: applies the SAME scale (no re-learning)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature scaling complete!")

# ==========================================================
# 6. BUILD THE ANN MODEL
# ==========================================================
print("\n🏗️  Building ANN model...")

model = Sequential([
    # First hidden layer
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),

    # Second hidden layer
    Dense(32, activation='relu'),
    Dropout(0.2),

    # Third hidden layer
    Dense(16, activation='relu'),

    # Output layer (linear activation for regression)
    Dense(1, activation='linear')
])

# Display model architecture summary
model.summary()

# ==========================================================
# 7. COMPILE THE MODEL
# ==========================================================
print("\n⚙️  Compiling the model...")
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mae']
)
print("✅ Model compiled!")

# ==========================================================
# 8. TRAIN THE MODEL
# ==========================================================
print("\n🚀 Training the model...")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,    # Use the best weights found
    verbose=1                     # Print when stopping
)

history = model.fit(
    X_train_scaled, Y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)
print("✅ Training complete!")

# ==========================================================
# 9. EVALUATE ON TEST DATA
# ==========================================================
print("\n📊 Evaluating on test data...")

# Make predictions on the test set
Y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

print(f"\n   Test Metrics:")
print(f"   MSE: {mse:.4f}")
print(f"   RMSE: {rmse:.4f}")
print(f"   R² Score: {r2:.4f}")

# ==========================================================
# 10. VISUALIZE TRAINING HISTORY
# ==========================================================
print("\n📈 Plotting training history...")

# Create 1 row, 2 columns of plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# ----- Plot 1: Loss (MSE) over Epochs -----
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.legend()
ax1.set_title('Loss over Epochs')
ax1.grid(True)

# ----- Plot 2: MAE over Epochs -----
ax2.plot(history.history['mae'], label='Training MAE')
ax2.plot(history.history['val_mae'], label='Validation MAE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend()
ax2.set_title('MAE over Epochs')
ax2.grid(True)

plt.tight_layout()
plt.savefig('saved_visualizations/admission_training_history.png')
plt.show(block=False)
plt.pause(2)
plt.close()
print("✅ Training history saved as 'saved_visualizations/admission_training_history.png'")

# ==========================================================
# 11. PREDICTION VS ACTUAL PLOT
# ==========================================================
print("\n📈 Plotting Predictions vs Actual...")

fig, ax = plt.subplots(figsize=(8, 6))

# Scatter plot of actual vs predicted
ax.scatter(Y_test, Y_pred, alpha=0.5, color='blue')

# Red dashed line for perfect predictions (y=x)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)

# Labels and title
ax.set_xlabel('Actual Chance of Admit')
ax.set_ylabel('Predicted Chance of Admit')
ax.set_title('Predictions vs Actual')
ax.grid(True, alpha=0.3)

# Save and display
plt.savefig('saved_visualizations/predictions_vs_actual.png')
plt.show()
print("✅ Predictions vs Actual plot saved as 'saved_visualizations/predictions_vs_actual.png'")

# ==========================================================
# 12. SAVE THE MODEL AND ARTIFACTS
# ==========================================================
print("\n💾 Saving the model...")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the trained model
model.save('models/admission_model.h5')
print("✅ Model saved as 'models/admission_model.h5'")

# Save the scaler
# Required to scale new input data the same way
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved as 'models/scaler.pkl'")

# Save feature names
feature_names = X.columns.tolist()
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✅ Feature names saved as 'models/feature_names.pkl'")

print("\n" + "=" * 60)
print("✅ ADMISSION PREDICTION TRAINING COMPLETE!")
print("=" * 60)
print(f"   R² Score: {r2:.4f}")
print(f"   RMSE: {rmse:.4f}")
print("\n📌 Next step:")
print("   Run: streamlit run app.py")
print("=" * 60)