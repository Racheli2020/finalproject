from flask import Flask, render_template, request
import pickle
import numpy as np
import secrets
import pandas as pd
from flask_wtf.csrf import CSRFProtect

secret_key = secrets.token_hex(16)
app = Flask(__name__)
app.secret_key = secret_key
csrf = CSRFProtect(app)



with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)


def n_formatter(num, digits):
    lookup = [
        {'value': 1, 'symbol': ''},
        {'value': 1e3, 'symbol': 'k'},
        {'value': 1e6, 'symbol': 'M'},
        {'value': 1e9, 'symbol': 'G'},
        {'value': 1e12, 'symbol': 'T'},
        {'value': 1e15, 'symbol': 'P'},
        {'value': 1e18, 'symbol': 'E'}
    ]
    rx = r'\.0+$|(\.[0-9]*[1-9])0+$'
    item = next((item for item in reversed(lookup) if num >= item['value']), None)
    if item:
        return f"{num / item['value']:.{digits}f}".rstrip('0').rstrip('.') + item['symbol']
    else:
        return '0'


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        dataToPredict = {}

        dataToPredict['City'] = request.form['city']
        dataToPredict['condition '] = request.form['condition']
        dataToPredict['room_number'] = request.form['room_number']
        dataToPredict['Area'] = request.form['Area']
        dataToPredict['furniture '] = request.form['furniture']
        dataToPredict['total_floors'] = request.form['total_floors']
        dataToPredict['hasMamad '] = request.form['hasMamad']
        dataToPredict['hasElevator '] = request.form['hasElevator']

        print(dataToPredict)

        dataToPredictDF = pd.DataFrame(dataToPredict, index=[0])

        print("-------------------------------")
        print(dataToPredictDF)

        # Run your model with the form data
        output = model.predict(dataToPredictDF)

        output = n_formatter(output[0], 4)

        # Pass the output to the template
        return render_template('index.html', output=output)

        # Render the initial form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5001)

