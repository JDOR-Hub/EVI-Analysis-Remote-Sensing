import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

evi_file = "output/EVI_Neiva.tif"
evi_labels = "output/labels_Neiva.tif"
blue_band = "data/Neiva-2024-04-10/SR_B2.TIF"
red_band = "data/Neiva-2024-04-10/SR_B4.TIF"
nir_band = "data/Neiva-2024-04-10/SR_B5.TIF"

# Load imagery data

with rasterio.open(evi_file) as src:
    evi_data = src.read(1)
    profile = src.profile

with rasterio.open(evi_labels) as src:
    evi_labels_data = src.read(1)

with rasterio.open(blue_band) as src:
    blue_data = src.read(1)

with rasterio.open(red_band) as src:
    red_data = src.read(1)

with rasterio.open(nir_band) as src:
    nir_data = src.read(1)


# Create a DataFrame for visualization
df = pd.DataFrame({
    'EVI': evi_data.flatten(),
    'Blue': blue_data.flatten(),
    'Red': red_data.flatten(),
    'NIR': nir_data.flatten(),
    'Water': evi_labels_data.flatten()
})

#print("DataFrame:\n", df)

df['Water'] = np.where(df['Water'] > 0, True, False)
#print(df.head())

'''descriptive = df.describe(percentiles=[0.01,.03,.05,.07,.08,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.98,.99])
print("Descriptive Statistics:\n", descriptive)'''

# Correlation matrix
'''correlation_matrix = df[['EVI', 'Blue', 'Red', 'NIR', 'Water']].corr()
print("Correlation Matrix:\n", correlation_matrix)'''


# Create figure with 2x2 subplots
'''fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Spectral Band Histograms', fontsize=16)

# Plot 1: EVI
axes[0, 0].hist(df['EVI'], bins=50, color='orange', alpha=0.7)
axes[0, 0].set_title('EVI Histogram')
axes[0, 0].set_xlabel('EVI Values')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid()

# Plot 2: Blue
axes[0, 1].hist(df['Blue'], bins=50, color='blue', alpha=0.7)
axes[0, 1].set_title('Blue Band Histogram')
axes[0, 1].set_xlabel('Blue Band Values')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid()

# Plot 3: Red
axes[1, 0].hist(df['Red'], bins=50, color='red', alpha=0.7)
axes[1, 0].set_title('Red Band Histogram')
axes[1, 0].set_xlabel('Red Band Values')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid()

# Plot 4: NIR
axes[1, 1].hist(df['NIR'], bins=50, color='gray', alpha=0.7)
axes[1, 1].set_title('NIR Band Histogram')
axes[1, 1].set_xlabel('NIR Band Values')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid()

plt.tight_layout()
plt.show()'''

'''n_sample =  300
sample =  pd.concat([
    df[df['Water'] == class_value].sample(n=n_sample, random_state=42, replace=False)
    for class_value in df['Water'].unique()
])

pair_plot = sns.pairplot(data=sample, hue="Water")
pair_plot.fig.suptitle("Feature Relationships Stratified by Water Class (n=300 per class)")
plt.show()'''

# Machine learning setup
# Set sample size per class for balanced sampling
sample_size_per_class = 10000  # Can be adjusted (e.g., 300000 for larger datasets)

# Create balanced dataset by sampling equal numbers from each class
df_sampled = pd.concat([
    df[df['Water'] == class_value].sample(n=sample_size_per_class, 
                                           random_state=42, 
                                           replace=False)
    for class_value in df['Water'].unique()
])

# Verify class distribution in sampled data
print("Class distribution in sampled dataset:")
print(df_sampled['Water'].value_counts())

# Prepare features (X) and target (y)
X = df_sampled.drop(['Water', 'Blue', 'EVI'], axis=1)  # Independent variables (features)
y = df_sampled['Water']                        # Dependent variable (target)

# Split data into training and test sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.1, 
    random_state=42, 
    stratify=y  # Maintain class distribution in splits
)

# Show first few rows of features
print("\nFeature preview:")
print(X.head())

# Verify training set class distribution
print("\nClass distribution in training set:")
print(y_train.value_counts())

# Initialize SVM model with linear kernel
svm_model = SVC(kernel='linear', random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions on test set
y_pred = svm_model.predict(X_test)

# Calculate model performance metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output results
print("\nPredicted classes:")
print(y_pred)
print("\nSVM model accuracy:", accuracy)
print("\nClassification report:\n", report)