import numpy as np
from flask import Flask, request, jsonify, render_template
import statistics
import joblib

app = Flask(__name__, static_folder="templates/assets")

# Load saved models

knn_model = joblib.load('models/knn.sav')
lr_model = joblib.load('models/logistic_regression.sav')
rf_model = joblib.load('models/random_forest.sav')


# Dictionary of all loaded models
loaded_models = {
    'knn': knn_model,
    'lr': lr_model,
    'rf': rf_model
}

# Function to decode predictions 
def decode(pred):
    if pred == 1: return 'We can conclude that The Customer will Exits'
    else: return 'We can conclude that The Customer will Stays'

@app.route('/')
def home():
    # Initial rendering
    result = [{'prediction': ' '}]
    
    # Create main dictionary
    maind = {}
    maind['customer'] = {}
    maind['predictions'] = result

    return render_template('index.html', maind=maind)

@app.route('/predict', methods=['POST'])
def predict():

    # List values received from index
    values = [x for x in request.form.values()]

    # new_array - input to models
    new_array = np.array(values).reshape(1, -1)
    print(new_array)
    print(values)
    
    # Key names for customer dictionary custd
    cols = ['CreditScore',
            'Gender',
            'Age',
            'Tenure',
            'Balance',
            'NumOfProducts',
            'HasCrCard',
            'IsActiveMember',
            'EstimatedSalary']

    # Create customer dictionary
    custd = {}
    for k, v in  zip(cols, values):
        custd[k] = v

    # Convert 1 or 0 to Yes or No    
    yn_val = ['HasCrCard', 'IsActiveMember']
    for val in  yn_val:
        if custd[val] == '1': custd[val] = 'Yes'
        else: custd[val] = 'No'

    # Loop through 'loaded_models' dictionary and
    # save predictiond to the list
    predl = []
    for m in loaded_models.values():
        predl.append(decode(m.predict(new_array)[0]))

    result = [{'prediction': statistics.mode(predl)}]

    # Create main dictionary
    maind = {}
    maind['customer'] = custd
    maind['predictions'] = result
    
    return render_template('index.html', maind=maind)

@app.route('/')
def nav():
    return render_template('index.html')

if __name__ == "__main__":
    app.run()
