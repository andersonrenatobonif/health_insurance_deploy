import os
import numpy as np
import pickle
import pandas as pd
from flask          import Flask, request, Response
from health.Health  import health

#loading model
path = ''
model = pickle.load(open(path + 'model/model_health.pkl', 'rb'))

app = Flask(__name__)

@app.route('/health/predict', methods=['POST'])


def health_predict():
    test_json = request.get_json()
    
    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
            
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            
            
        #instance health class
        
        pipeline = health()
        
        #df1 = pipeline.data_cleaning(test_raw)
        
        #df2 = pipeline.feature_engineering(df1)
        
        df2 = pipeline.data_preparation(test_raw)
        
        df_response = pipeline.get_predictions(model, test_raw, df2)
        
        return df_response
    
    else:
    
    	return Response( '{}', status=200, mimetype='application/json' )

if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000)
    app.run(host = '0.0.0.0', port = port)
