import pandas as pd
import pickle
import numpy as np


class health (object):
    def __init__(self):
        self.home_path              = ''
        self.age_scaler             = pickle.load(open(self.home_path + 'parameter/age_scaler.pkl'      , 'rb'))
        self.vintage_scaler         = pickle.load(open(self.home_path + 'parameter/vintage_scaler.pkl'  , 'rb'))
        self.vehicle_age_encoding   = pickle.load(open(self.home_path + 'parameter/vehicle_age.pkl'     , 'rb'))
        self.time_customer_encoding = pickle.load(open(self.home_path + 'parameter/time_of_customer.pkl', 'rb'))
        self.vehicle_damage         = pickle.load(open(self.home_path + 'parameter/vehicle_damage.pkl'  , 'rb'))
        self.annual_premium_scaler  = pickle.load(open(self.home_path + 'parameter/annual_premium.pkl'  , 'rb'))
        return
    
    
    def data_cleaning(self, df):
        cols =  ['id', 'gender', 'age', 'region_code', 'policy_sales_channel',
                 'previously_insured', 'annual_premium', 'vintage', 'driving_license',
                 'vehicle_age', 'vehicle_damage']
        
        df.columns = cols
        
        # df['time_of_customer'] = (df['vintage'].apply(lambda x: 'new' if x < 50 else 'moderate' 
        #                                                if x < 100 else 'medium' 
        #                                                if x < 150 else 'professional' 
        #                                                if x < 200 else 'expert' 
        #                                                if x < 250 else 'old'))
        
        
        
        return df
        
        
   
    
    def data_preparation(self, df1):
        
        # age
        df1['age'] =  self.age_scaler.transform(df1[['age']].values)
        
        # vintage
        df1['vintage'] = self.vintage_scaler.transform(df1[['vintage']].values)
        
        #gender val
        df1['gender'] = df1['gender'].apply(lambda x: 0 if x == 'Female' else 1)
        
        #region_code
        #target_region_code = df1.groupby('region_code')['response'].mean()
        target_region_code = df1.groupby('region_code').size() / len(df1)
        df1.loc[:, 'region_code'] = df1['region_code'].map(target_region_code)
        
        # vehicle_age - Order Encoding
        #df1['vehicle_age'] = self.vehicle_age_encoding.transform(df1['vehicle_age'])
        df1['vehicle_age'] = df1['vehicle_age'].apply(lambda x: '1' if x == '< 1 Year' else '2' if x == '1-2 Year' else '3')
        
        # time_of_customer - Label
        #df1['time_of_customer'] = self.time_customer_encoding.transform(df1['time_of_customer'])
        
        # vehicle_damage - Label
        df1['vehicle_damage'] = df1['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # policy_sales_channel - Frequency
        fe_policy_channel = df1.groupby('policy_sales_channel').size() / len(df1)
        df1.loc[:, 'policy_sales_channel'] = df1['policy_sales_channel'].map(fe_policy_channel)
        
        # annual_premium
        df1['annual_premium'] = self.annual_premium_scaler.transform(df1[['annual_premium']].values)
        
        #fillna
        df1 = df1.fillna(0)
        
        cols_selected = ['age', 'region_code', 'policy_sales_channel', 'vehicle_damage',
                         'previously_insured', 'annual_premium', 'vintage']
        
        return df1[cols_selected]
    
    
    
    def get_predictions(self, model, original_data, test_data):
        
        
        pred = model.predict_proba(test_data)
        
        original_data['score'] = pred[:, 1].tolist()
        
        return original_data.to_json(orient='records', date_format='iso')