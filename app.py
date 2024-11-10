import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
model = pickle.load(open('modelgbt.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     int_features  = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)
#     output = round(prediction[0],2)
#     return render_template('home.html',prediction_text="Number of Weekly Rides Should be {}".format(math.floor(output)))

@app.route('/predict', methods=['POST'])
def predict():
    int_features  = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    if prediction[0] == 1:
        output_text = "Tubuh Anda terdeteksi diabetes"
    else:
        output_text = "Tubuh Anda tidak terdeteksi diabetes."
    
    return render_template('home.html', prediction_text=output_text)


if __name__ == '__main__':
    app.run()
