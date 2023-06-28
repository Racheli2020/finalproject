from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        dataToPredict = {}

        dataToPredict['type'] = request.form['type']
        dataToPredict['City'] = request.form['city']
        dataToPredict['condition '] = request.form['condition']
        dataToPredict['room_number'] = request.form['room_number']
        dataToPredict['Area'] = request.form['Area']
        dataToPredict['furniture '] = request.form['furniture']
        dataToPredict['total_floors'] = request.form['total_floors']
        dataToPredict['hasMamad '] = request.form['hasMamad']
        dataToPredict['hasElevator '] = request.form['hasElevator']


        dataToPredictDF = pd.DataFrame(dataToPredict, index=[0])


        # Run your model with the form data
        output = model.predict(dataToPredictDF)


        # Pass the output to the template
        print(request.method)
        return render_template('index.html', output=output)
    # Render the initial form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5001)

