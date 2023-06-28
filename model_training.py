from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from madlan_data_prep import prepare_data


data = pd.read_excel('C:/Users/shake/.jupyter/EX5/output_all_students_Train_v10.xlsx', engine='openpyxl')
Data = prepare_data(data)


# Define selected features
selected_features = [ 'City', 'condition ', 'room_number', 'Area', 'furniture ', 'total_floors', 'hasMamad ', 'hasElevator ']

# Split the data into features and target
X = Data[selected_features]
y = Data['price']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)

# Define numerical and categorical columns
num_cols = [col for col in X_train.columns if X_train[col].dtypes != 'O']
cat_cols = [col for col in X_train.columns if X_train[col].dtypes == 'O']

X_train[cat_cols] = X_train[cat_cols].astype(str)

# Define preprocessing pipelines
numerical_pipeline = Pipeline([
    ('numerical_imputation', SimpleImputer(strategy='median', add_indicator=False)),
    ('scaling', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('categorical_imputation', SimpleImputer(strategy='constant', add_indicator=False, fill_value='missing')),
    ('one_hot_encoding', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# Define the column transformer
column_transformer = ColumnTransformer([
    ('numerical_preprocessing', numerical_pipeline, num_cols),
    ('categorical_preprocessing', categorical_pipeline, cat_cols)
], remainder='passthrough')

# Build the preprocessing and modeling pipeline
pipe_preprocessing_model = Pipeline([
    ('preprocessing_step', column_transformer),
    ('model', ElasticNet(alpha=0.001, l1_ratio= 0.000000001))  # Change the value of alpha here
])


# Perform 10-fold cross-validation and evaluate metrics
cv_scores = cross_val_score(pipe_preprocessing_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
mse_scores = -cv_scores
mean_mse = mse_scores.mean()

r2_scores = cross_val_score(pipe_preprocessing_model, X_train, y_train, cv=10, scoring='r2')

average_mse = mse_scores.mean()
average_r2 = r2_scores.mean()
rmse = np.sqrt(average_mse)

# Fit the model on the training data
pipe_preprocessing_model.fit(X_train, y_train)

# Predict on the test data
y_pred = pipe_preprocessing_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Training Set:")
print('Mean Squared Error:', mean_mse)
print(" R^2:", average_r2)
print("RMSE:", rmse)
print("\nTest Set:")
print("MSE:", mse)
print("R^2:", r2)




import pickle
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(pipe_preprocessing_model, file)
