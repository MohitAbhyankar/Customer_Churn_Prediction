import pandas as pd 
from flask import Flask,jsonify,request
import joblib

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    req = request.get_json()
    input_data = req['data']
    input_data_df = pd.DataFrame.from_dict(input_data)

    model = joblib.load('model_2.pkl')

    scale_obj = joblib.load('scale.pkl')

    input_data_scaled = scale_obj.transform(input_data_df)

    print(input_data_scaled)

    predict = model.predict(input_data_df)

    if predict[0] == 0.0:
        churn = 'No'
    
    else:
        churn = 'Yes'
    
    return jsonify({'output':{'Customer Churned':churn}})

@app.route('/')
def home():
    return 'Welcome to churn prediction app'

if __name__=='__main__':
    app.run(host='0.0.0.0',port='3000')