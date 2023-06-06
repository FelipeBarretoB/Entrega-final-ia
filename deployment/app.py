from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)

model = pickle.load(open('kn_model.sav', 'rb'))


@app.route('/')
def hello_world():
    print(type(model))
    return render_template("home.html")


@app.route('/predict',methods=['POST'])
def predict():
    
    features = [x for x in request.form.values()]
    print(features)
    columns=['Name','Platform','Year','Genre','Publisher','YearsSincePublished']
    features.append(2020-float(features[2]))
    features[2]=float(features[2])
    print(len(features))
    input_df = pd.DataFrame([features],columns=columns)
    result = model.predict(input_df)
    return render_template("home.html", pred="Las ventas esperadas son: {}".format(result))

if __name__ == '__main__':
    app.run(host='0.0.0.0')