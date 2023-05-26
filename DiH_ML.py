import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef

# Load the data
train_data = pd.read_csv('hospital_deaths_train.csv')
test_data = pd.read_csv('hospital_deaths_test.csv')

train_data.drop(['recordid'], axis=1, inplace=True)
test_data.drop(['recordid'], axis=1, inplace=True)

# Concatenate and shuffle the data
combined_data = pd.concat([train_data, test_data], axis=0)
shuffled_data = combined_data.sample(frac=1, random_state=42)

# Split the data into train and test sets
train_ratio = len(train_data) / len(combined_data)
train_data, test_data = train_test_split(shuffled_data, train_size=train_ratio, random_state=42)


# Preprocessing
scaler = StandardScaler()
imputer = SimpleImputer(strategy='mean')

X_train = train_data.drop(['In-hospital_death'], axis=1)
X_train = scaler.fit_transform(X_train)
X_train = imputer.fit_transform(X_train)
y_train = train_data['In-hospital_death']

X_test = test_data.drop(['In-hospital_death'], axis=1)
X_test = scaler.transform(X_test)
X_test = imputer.transform(X_test)
y_test = test_data['In-hospital_death']

# Convert to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train.values.reshape(-1, 1), dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test.values.reshape(-1, 1), dtype=tf.float32)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Predict probabilities on test data
y_pred = model.predict(X_test)

# Convert probabilities to binary predictions
y_pred_binary = np.where(y_pred > 0.35, 1, 0)

# Calculate MCC score
mcc_score = matthews_corrcoef(y_test, y_pred_binary)
print(f'Test MCC score: {mcc_score:.6f}')