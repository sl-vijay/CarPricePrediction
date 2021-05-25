from flask import Flask, request
import pickle
import numpy as np
import pandas as pd
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return "Welcome All"


@app.route("/predict", methods=['GET'])
def predict_car_price():
    
    """Lets try to predict Car Price
    This is using docstrings for specifications
    ---
    parameters:
        
      - name: Present_Price
        in: query
        type: number
        required: true
      - name: Kms_Driven
        in: query
        type: number
        required: true
      - name: Owner
        in: query
        type: number
        required: true
      - name: Age_Year
        in: query
        type: number
        required: true
      - name: Fuel_Type_Diesel
        in: query
        type: number
        required: true
      - name: Fuel_Type_Petrol
        in: query
        type: number
        required: true
      - name: Seller_Type_Individual
        in: query
        type: number
        required: true
      - name: Transmission_Manual
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    """
    
    Present_Price=request.args.get("Present_Price")
    Kms_Driven=request.args.get("Kms_Driven")
    Owner=request.args.get("Owner")
    Year=request.args.get("Age_Year")
    Fuel_Type_Diesel=request.args.get("Fuel_Type_Diesel")
    Fuel_Type_Petrol=request.args.get("Fuel_Type_Petrol")
    Seller_Type_Individual=request.args.get("Seller_Type_Individual")
    Transmission_Manual=request.args.get("Transmission_Manual")
    Present_Price=request.args.get("Present_Price")
    Present_Price=request.args.get("Present_Price")
    prediction=model.predict([[Present_Price,Kms_Driven,Owner,Year,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Manual]])
    output=round(prediction[0],2)
    return "You can sell the car for "+str(output)+" Lakhs"


if __name__=="__main__":
    app.run(host = "127.0.0.1", port = 5000, debug=False)