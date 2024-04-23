import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load the machine learning models
svc = SVC()  # Example, replace with your actual SVC model
rf = RandomForestClassifier()  # Example, replace with your actual Random Forest model
gb = GradientBoostingClassifier()  # Example, replace with your actual Gradient Boosting model

# Load the pickled models
with open('svc_stress.pkl', 'wb') as file:
    pickle.dump(svc, file, protocol=pickle.HIGHEST_PROTOCOL)

with open('rf_stress.pkl', 'wb') as file:
    pickle.dump(rf, file, protocol=pickle.HIGHEST_PROTOCOL)

with open('gradient_stress.pkl', 'wb') as file:
    pickle.dump(gb, file, protocol=pickle.HIGHEST_PROTOCOL)

# Load the dataset
data = pd.read_csv(r'C:\Users\yerra\Downloads\employee\employee\CODING\FRONTEND\stress_test.csv')

def predict(algo, row):
    test_data = data.iloc[row].values.reshape(1, -1)
    if algo == 'svc':
        y_pred = svc.predict(test_data)
    elif algo == 'rf':
        y_pred = rf.predict(test_data)
    else:
        y_pred = gb.predict(test_data)
    return y_pred


