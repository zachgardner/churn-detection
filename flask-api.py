import numpy as np
import pandas as pd
from flask import Flask, abort, request, jsonify
import json
from pandas.io.json import json_normalize
import _pickle as cPickle
import sklearn

#Open serialized model definition using Pickle functionality in python
file = open("model.pkl","rb")
model = cPickle.load(file)

#Open serialized column data frame definition using Pickle functionality in python
columns = open("columns.pkl","rb")
columns_list = cPickle.load(columns)

#Intialize Flash implementation to host API
app = Flask(__name__)

#Define route configutation to model function
@app.route('/api', methods=['POST'])
def churnPrediction():
    
    #Retrieve data payload from incoming HTTP Post Eequest 
    content = request.get_data()
    #Convert incoming JSON payload data into pandas series
    X_predict = pd.read_json(content, typ='series')
    #Assemble a data frame that matches definition from serialized python column list
    df = pd.DataFrame(columns=columns_list)
    #Add pandas series to the assembled data frame 
    df = df.append(X_predict,ignore_index=True)
    #Predict the 1 or 0 response form the Gradient Boost churn risk model.
    churn_prediction = model.predict(df)
    output = churn_prediction[0]
    #Return the churn prediction via JSON as HTTP Response.
    prediction_response = '{"result":' + str(output) + '}'
    return(prediction_response)



@app.route('/', methods=['GET', 'POST'])
def hi():
    return("hi")

if __name__ == '__main__':
    app.run()