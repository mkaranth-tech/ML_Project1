import numpy as np
from flask import Flask, jsonify,request,render_template
from pandas import DataFrame
import pickle

# load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

# routes
@app.route('/predict', methods=['POST'])

def predict():
    # get data
    output=''
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final_features=[np.array(int_features)]
    # predictions
    result = model.predict(final_features)
    comp=result[0]
    if comp==1:
        output='alive'
    else:
        output='dead'

    # return data
    return render_template('index.html',prediction_text='predicted output: Individual is {}'.format(output))

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
