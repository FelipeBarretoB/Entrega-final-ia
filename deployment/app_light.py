from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
import os
app = Flask(__name__)

model = pickle.load(open('kn_model.sav', 'rb'))


@app.route('/')
def hello_world():
    return render_template("home.html")


@app.route('/predict',methods=['POST'])
def predict():
    
    features = [x for x in request.form.values()]
    columns=['Name','Platform','Year','Genre','Publisher','YearsSincePublished']
    features.append(2020-float(features[2]))
    features[2]=float(features[2])
    input_df = pd.DataFrame([features],columns=columns)
    result = model.predict(input_df)
    return render_template("home.html", pred="Las ventas esperadas son: {}".format(result))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(debug=True, host='0.0.0.0', port=port)