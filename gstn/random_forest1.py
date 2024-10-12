#**Acuracy, Precision, Recall, F1score for RANDOM FOREST**
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Load the datasets
X_train = pd.read_csv('/content/X_Train_Data_Input.csv')
Y_train = pd.read_csv('/content/Y_Train_Data_Target.csv')
X_test = pd.read_csv('/content/X_Test_Data_Input.csv')
Y_test = pd.read_csv('/content/Y_Test_Data_Target.csv')

# ---Ensure that X_train and Y_train have the same number of rows before concatenation---
X_train = X_train.iloc[:min(X_train.shape[0], Y_train.shape[0])]
Y_train = Y_train.iloc[:min(X_train.shape[0], Y_train.shape[0])]
X_test = X_test.iloc[:min(X_test.shape[0], Y_test.shape[0])]
Y_test = Y_test.iloc[:min(X_test.shape[0], Y_test.shape[0])]

# ---Extract the target variable from Y_train and Y_test---
y_train = Y_train['target']
y_test = Y_test['target']

# ---Now you can split the data using X_train, y_train---
# Split data into training and testing sets (if necessary, you might skip this step)
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Handle non-numeric columns (drop for now)
X_train = X_train.select_dtypes(include=['number'])
X_test = X_test.select_dtypes(include=['number'])

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")