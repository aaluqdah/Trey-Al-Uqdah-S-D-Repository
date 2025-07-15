def main():
    #!/usr/bin/env python
    # coding: utf-8

    # In[1]:


    import requests
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    import matplotlib.pyplot as plt
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error


    # In[2]:


    # Pull EIA Data
    EIA_KEY = 'S5kBDUycUiKkFCG6uxywpfYtq7IzU7AAhOgoUU4y'


    # Initialize parameters
    Date = "2010-01-01"
    End = None
    # API endpoints 
    # EIA endpoints

    url = 'https://api.eia.gov/v2/petroleum/sum/snd/data/'   
    urlTotal = 'https://api.eia.gov/v2/total-energy/data/'

    # Parameters

    Stocks = {
        "api_key": EIA_KEY, # API key
        "frequency": "monthly", # Frequency of data
        "data[0]": "value",  
        "facets[msn][0]": "COSXPUS",  # Series ID
        "start": Date , # Start date
        "end": End, # End date
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 500
    }

    StockChange = {
        "api_key": EIA_KEY, # API key
        "frequency": "monthly", # Frequency of data
        "data[0]": "value",  
        "facets[series][0]": "MCRSCUS2",  # Series ID
        "start": Date , # Start date
        "end": End, # End date
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 500
    }


    response2 = requests.get(urlTotal, params=Stocks)
    data2 = response2.json()
    df6 = pd.DataFrame(data2['response']['data'])
    df6 = df6[['period', 'value']]
    df6.rename(columns={'value': 'Stocks'}, inplace=True)
    df6.rename(columns={'period': 'ds'}, inplace=True)
    df6['ds'] = pd.to_datetime(df6['ds'])
    df6['Stocks'] = pd.to_numeric(df6['Stocks'], errors='coerce')
    df_merged = df6

    response2 = requests.get(url, params=StockChange)
    data2 = response2.json()
    df7 = pd.DataFrame(data2['response']['data'])
    df7 = df7[['period', 'value']]
    df7.rename(columns={'value': 'Stock Change'}, inplace=True)
    df7.rename(columns={'period': 'ds'}, inplace=True)
    df7['ds'] = pd.to_datetime(df7['ds'])
    df7['Stock Change'] = pd.to_numeric(df7['Stock Change'], errors='coerce')
    df_merged = pd.merge(df_merged, df7, on='ds', how='outer')

    df = pd.read_pickle("ProductionS.pkl")
    df = df[['ds','Prod']]
    df['ds'] = pd.to_datetime(df['ds'])
    df['Prod'] = pd.to_numeric(df['Prod'], errors='coerce')  # Ensure numeric
    df_merged = pd.merge(df_merged, df, on='ds', how='outer')

    df = pd.read_pickle("ImportS.pkl")
    df = df[['ds','Imports']]
    df['ds'] = pd.to_datetime(df['ds'])
    df['Imports'] = pd.to_numeric(df['Imports'], errors='coerce')  # Ensure numeric
    df_merged = pd.merge(df_merged, df, on='ds', how='outer')


    df = pd.read_pickle("RefineryD.pkl")
    df = df[['ds','Demand']]
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.rename(columns={'Demand': 'Refinery'})
    df_merged = pd.merge(df_merged, df, on='ds', how='outer')

    df = pd.read_pickle("ExportD.pkl")
    df = df[['ds','Exports']]
    df['ds'] = pd.to_datetime(df['ds'])
    df_merged = pd.merge(df_merged, df, on='ds', how='outer')



    df_merged['Demand'] = df_merged['Refinery'] + df_merged['Exports']
    df_merged['Supply'] = df_merged['Prod'] + df_merged['Imports'] 


    df_merged.to_pickle("S&D.pkl")
    print(df_merged.tail(12))

    df_merged['Spread'] = df_merged['Supply'] - df_merged['Demand']


    # Pull EIA Data
    EIA_KEY = 'S5kBDUycUiKkFCG6uxywpfYtq7IzU7AAhOgoUU4y'


    # Initialize parameters
    Date = "2010-01-01"
    End = None
    # API endpoints 
    # EIA endpoints


 
    urlPrice = 'https://api.eia.gov/v2/petroleum/pri/spt/data/'   

    WTI = {
        "api_key": EIA_KEY, # API key
        "frequency": "monthly", # Frequency of data
        "data[0]": "value",  
        "facets[series][0]": "RWTC",  # Series ID
        "start": Date , # Start date
        "end": End, # End date
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 500
    }

    response2 = requests.get(urlPrice, params=WTI)
    data2 = response2.json()
    df6 = pd.DataFrame(data2['response']['data'])
    df6 = df6[['period', 'value']]
    df6.rename(columns={'value': 'WTI'}, inplace=True)
    df6.rename(columns={'period': 'ds'}, inplace=True)

    # Ensure datetime format for merge
    df6['ds'] = pd.to_datetime(df6['ds'])
    df_merged['ds'] = pd.to_datetime(df_merged['ds'])

    # Merge on 'ds' (outer to preserve all rows, or 'inner' for overlap only)
    df_merged = pd.merge(df_merged, df6, on='ds', how='outer')

    # Optional: sort by date and reset index
    df_merged2 = df_merged.sort_values('ds').reset_index(drop=True)
    df_merged2.to_csv("df_merged2.csv", index=False)
    print(df_merged2.tail(20))



if __name__ == "__main__":
    main()
