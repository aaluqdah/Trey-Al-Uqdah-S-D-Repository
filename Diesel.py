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


    # Pull FRED Data
    FRED_KEY = '02b2c2d2bbf883773e270d39679071c9'
    FREDurl = 'https://api.stlouisfed.org/fred/series/observations'



    param1 = {
        'series_id': 'TRUCKD11',
        'api_key': FRED_KEY,
        'file_type': 'json'}

    response = requests.get(FREDurl, params=param1)
    data = response.json()
    observations = data['observations']
    df1 = pd.DataFrame(observations)
    df1 = df1.rename(columns={"date": "ds","value": "TONS"})
    df1 = df1[['ds', 'TONS']]




    param2 = {
        'series_id': 'HTRUCKSSAAR',
        'api_key': FRED_KEY,
        'file_type': 'json'}

    response = requests.get(FREDurl, params=param2)
    data = response.json()
    observations = data['observations']
    df2 = pd.DataFrame(observations)
    df2 = df2.rename(columns={"date": "ds","value": "TRUCKS"})
    df2 = df2[['ds', 'TRUCKS']]



    param3 = {
        'series_id': 'INDPRO',
        'api_key': FRED_KEY,
        'file_type': 'json'}

    response = requests.get(FREDurl, params=param3)
    data = response.json()
    observations = data['observations']
    df3 = pd.DataFrame(observations)
    df3 = df3.rename(columns={"date": "ds","value": "IND"})
    df3 = df3[['ds', 'IND']]

    param4 = {
        'series_id': 'RAILFRTINTERMODALD11',
        'api_key': FRED_KEY,
        'file_type': 'json'}

    response = requests.get(FREDurl, params=param4)
    data = response.json()
    observations = data['observations']
    df7 = pd.DataFrame(observations)
    df7 = df7.rename(columns={"date": "ds","value": "RAIL"})
    df7 = df7[['ds', 'RAIL']]


    # Pull EIA Data
    EIA_KEY = 'S5kBDUycUiKkFCG6uxywpfYtq7IzU7AAhOgoUU4y'


    # Initialize parameters
    Date = "2010-01-01"
    End = None
    # API endpoints 
    # EIA endpoints


    url_WTI = 'https://api.eia.gov/v2/petroleum/pri/spt/data/'   
    totalurl = 'https://api.eia.gov/v2/total-energy/data/'

    # Parameters


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

    Diesel = {
        "api_key": EIA_KEY, # API key
        "frequency": "monthly", # Frequency of data
        "data[0]": "value",  
        "facets[series][0]": "EER_EPD2DXL0_PF4_Y35NY_DPG",  # Series ID
        "start": Date , # Start date
        "end": End, # End date
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 500
    }

    SALES = {
        "api_key": EIA_KEY, # API key
        "frequency": "monthly", # Frequency of data
        "data[0]": "value",  
        "facets[msn][0]": "DFACPUS",  # Series ID
        "start": Date , # Start date
        "end": End, # End date
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 500
    }



    response2 = requests.get(url_WTI, params=WTI)
    data2 = response2.json()
    df6 = pd.DataFrame(data2['response']['data'])
    df6 = df6[['period', 'value']]
    df6.rename(columns={'value': 'WTI'}, inplace=True)
    df6.rename(columns={'period': 'ds'}, inplace=True)

    response2 = requests.get(url_WTI, params=Diesel)
    data2 = response2.json()
    df4 = pd.DataFrame(data2['response']['data'])
    df4 = df4[['period', 'value']]
    df4.rename(columns={'value': 'Diesel'}, inplace=True)
    df4.rename(columns={'period': 'ds'}, inplace=True)

    response2 = requests.get(totalurl, params=SALES)
    data2 = response2.json()
    df5 = pd.DataFrame(data2['response']['data'])
    df5 = df5[['period', 'value']]
    df5.rename(columns={'value': 'SALES'}, inplace=True)
    df5.rename(columns={'period': 'ds'}, inplace=True)


    for df in [df1, df2, df3, df4, df5, df6, df7]:
        df['ds'] = pd.to_datetime(df['ds']).dt.to_period('M').dt.to_timestamp()
    df_merged = df6
    df_merged = pd.merge(df_merged, df4, on='ds', how='outer')
    df_merged = pd.merge(df_merged, df1, on='ds', how='outer')
    df_merged = pd.merge(df_merged, df2, on='ds', how='outer')
    df_merged = pd.merge(df_merged, df3, on='ds', how='outer')
    df_merged = pd.merge(df_merged, df7, on='ds', how='outer')
    df_merged = pd.merge(df_merged, df5, on='ds', how='outer')
    df_merged['month_num'] = df_merged['ds'].dt.month



    print(df_merged.tail(20))

    df_merged.to_pickle("Diesel.pkl")


    # In[3]:


    # Approximate current month pricing data
    df = pd.read_pickle("Diesel.pkl")
    df['ds'] = pd.to_datetime(df['ds'])


    # Pull EIA Data
    EIA_KEY = 'S5kBDUycUiKkFCG6uxywpfYtq7IzU7AAhOgoUU4y'

    urlPrice = 'https://api.eia.gov/v2/petroleum/pri/spt/data/' 
    # Initialize parameters
    Date = "2025-01-01"
    End = None
    # API endpoints 


    Diesel = {
        "api_key": EIA_KEY, # API key
        "frequency": "daily", # Frequency of data
        "data[0]": "value",  
        "facets[series][0]": "EER_EPD2DXL0_PF4_Y35NY_DPG",  # Series ID
        "start": Date , # Start date
        "end": End, # End date
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 500
    }

    response = requests.get(urlPrice, params=Diesel)
    data = response.json()
    dfdaily = pd.DataFrame(data['response']['data'])
    dfdaily = dfdaily[['period', 'value']]
    dfdaily.rename(columns={'value': 'Diesel'}, inplace=True)
    dfdaily.rename(columns={'period': 'ds'}, inplace=True)



    # Filter last 30 days and compute average
    # Ensure 'date' is datetime
    dfdaily['date'] = pd.to_datetime(dfdaily['ds'])
    dfdaily['Diesel'] = pd.to_numeric(dfdaily['Diesel'], errors='coerce')

    # Filter past 30 days
    today = datetime.today()
    start_date = today - timedelta(days=30)
    last_month_data = dfdaily[dfdaily['date'] >= start_date]

    # Compute average gas price
    D_avg = last_month_data['Diesel'].mean()


    # === Step 1: Get current month start ===
    last_reported_month = df['ds'].max()
    next_month_start = last_reported_month + relativedelta(months=1)




    # === Step 2: Check if row exists for current month ===
    if next_month_start not in df['ds'].values:
    
        # === Step 3: Get 30-day average prices from yfinance ===
        today = datetime.today()
        start_date = today - timedelta(days=30)

        wti = yf.download("CL=F", start=start_date, end=today)['Close'].dropna()
        wti_avg = wti.mean().item()

        # === Step 4: Create new row ===
        new_row = {
            'ds': next_month_start,
            'WTI': wti_avg,
            'Diesel': D_avg,
            # Fill other columns with NaN
            'IND': None,
            'TRUCKS': None,
            'TONS': None,
            'RAIL': None,
            'SALES': None,
        }

        # === Step 5: Append to DataFrame ===
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df['month_num'] = df['ds'].dt.month



    df.to_pickle("Diesel1.pkl")

    print(df.tail(20))


    # In[ ]:


    # === 1. SET PARAMETERS ===
    TARGET_COL = 'TONS'
    EXOG_COLS = ['WTI', 'Diesel']
    df = pd.read_pickle("Diesel1.pkl")  # Replace with your actual input file
    df = df[df['ds'] >= pd.to_datetime('2015-01-01')].reset_index(drop=True)

    # === 2. CLEAN DATA ===
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    for col in EXOG_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values('ds').reset_index(drop=True)

    # === 3. FEATURE ENGINEERING ===
    df['month_num'] = df['ds'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    for lag in [1, 2, 3, 6]:
        df[f'{TARGET_COL}_lag{lag}'] = df[TARGET_COL].shift(lag)
        for col in EXOG_COLS:
            df[f'{col.replace(" ", "")}_lag{lag}'] = df[col].shift(lag)

    df[f'{TARGET_COL}_roll3'] = df[TARGET_COL].rolling(window=3).mean().shift(1)
    for col in EXOG_COLS:
        df[f'{col.replace(" ", "")}_roll3'] = df[col].rolling(window=3).mean().shift(1)

    # === 4. DEFINE FEATURES & TARGET ===
    features = (
        EXOG_COLS +
        ['month_num', 'month_sin', 'month_cos'] +
        [f'{TARGET_COL}_lag{l}' for l in [1,2,3,6]] +
        [f'{TARGET_COL}_roll3'] +
        [f'{col.replace(" ", "")}_lag1' for col in EXOG_COLS] +
        [f'{col.replace(" ", "")}_lag2' for col in EXOG_COLS] +
        [f'{col.replace(" ", "")}_roll3' for col in EXOG_COLS]
    )

    df_hist = df.dropna(subset=[TARGET_COL] + features).copy()
    X_train = df_hist[features].apply(pd.to_numeric, errors='coerce')
    y_train = pd.to_numeric(df_hist[TARGET_COL], errors='coerce')

    valid_idx = X_train.dropna().index
    X_train = X_train.loc[valid_idx]
    y_train = y_train.loc[valid_idx]

    # === 5. TRUE HOLD-OUT BACKTEST (80% train, 20% test) ===
    df_backtest = df_hist.loc[valid_idx].copy()
    X_all = df_backtest[features]
    y_all = df_backtest[TARGET_COL]

    split_index = int(len(X_all) * 0.8)
    X_train_split, X_test = X_all.iloc[:split_index], X_all.iloc[split_index:]
    y_train_split, y_test = y_all.iloc[:split_index], y_all.iloc[split_index:]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=1,
        colsample_bytree=0.6,
        random_state=42
    )
    model.fit(X_train_split, y_train_split, verbose=False)

    y_pred_test = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    print("\n=== Hold-Out Backtest Error Metrics ===")
    print(f"MAE  (Mean Absolute Error):      {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"MAPE (Mean Absolute % Error):   {mape:.2f}%")

    # === 6. RE-TRAIN ON ALL DATA BEFORE FORECASTING ===
    model.fit(X_train, y_train, verbose=False)

    # === 7. FORECAST MISSING TARGET VALUES ===
    missing_idxs = df.index[df[TARGET_COL].isna()]

    for i in missing_idxs:
        prior = df.loc[:i - 1].copy()
        last = prior.iloc[-1]

        new_feats = {
            'month_num': df.at[i, 'ds'].month,
            'month_sin': np.sin(2 * np.pi * df.at[i, 'ds'].month / 12),
            'month_cos': np.cos(2 * np.pi * df.at[i, 'ds'].month / 12),
            f'{TARGET_COL}_lag1': last[TARGET_COL],
            f'{TARGET_COL}_lag2': last[f'{TARGET_COL}_lag1'],
            f'{TARGET_COL}_lag3': last[f'{TARGET_COL}_lag2'],
            f'{TARGET_COL}_lag6': prior[TARGET_COL].iloc[-6] if len(prior) >= 6 else last[TARGET_COL],
            f'{TARGET_COL}_roll3': pd.to_numeric(prior[TARGET_COL].iloc[-3:], errors='coerce').mean(),
        }

        for col in EXOG_COLS:
            new_feats[col] = df.at[i, col]
            new_feats[f'{col.replace(" ", "")}_lag1'] = last[col]
            new_feats[f'{col.replace(" ", "")}_lag2'] = last[f'{col.replace(" ", "")}_lag1']
            new_feats[f'{col.replace(" ", "")}_roll3'] = pd.to_numeric(prior[col].iloc[-3:], errors='coerce').mean()

        input_df = pd.DataFrame([new_feats])
        input_df = input_df.reindex(columns=features).apply(pd.to_numeric, errors='coerce')
        y_pred = model.predict(input_df)[0]

        df.loc[i, TARGET_COL] = y_pred
        df.loc[i, f'{TARGET_COL}_lag1'] = y_pred
        df.loc[i, f'{TARGET_COL}_lag2'] = last[f'{TARGET_COL}_lag1']
        df.loc[i, f'{TARGET_COL}_lag3'] = last[f'{TARGET_COL}_lag2']
        df.loc[i, f'{TARGET_COL}_lag6'] = prior[TARGET_COL].iloc[-6] if len(prior) >= 6 else y_pred

        roll_vals = pd.concat([prior[TARGET_COL].iloc[-2:], pd.Series([y_pred])])
        df.loc[i, f'{TARGET_COL}_roll3'] = roll_vals.mean() if len(roll_vals) == 3 else y_pred

        for col in EXOG_COLS:
            df.loc[i, f'{col.replace(" ", "")}_lag1'] = last[col]
            df.loc[i, f'{col.replace(" ", "")}_lag2'] = last[f'{col.replace(" ", "")}_lag1']
            df.loc[i, f'{col.replace(" ", "")}_roll3'] = pd.to_numeric(prior[col].iloc[-3:], errors='coerce').mean()

    # === 8. FINAL OUTPUT ===
    df = df[['ds', TARGET_COL]].copy()
    print("\n=== Final df with historical + forecasted values ===")
    print(df.tail(8).to_string(index=False))

    df_full = df

    #=============================#
    #    Optimize Parameters      #
    #=============================#

    """from sklearn.model_selection import GridSearchCV

    # === 5A. HYPERPARAMETER TUNING ===
    # === HYPERPARAMETER SEARCH ONLY ===
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 600, 800],
        'max_depth': [2, 4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.9, 0.8, 0.6],
        'colsample_bytree': [1, 0.8, 0.6]
    }

    grid_model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=grid_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\n=== Best Parameters Found ===")
    print(grid_search.best_params_)"""


    # In[ ]:


    # === 1. SET PARAMETERS ===
    TARGET_COL = 'TRUCKS'
    EXOG_COLS = ['WTI', 'Diesel']
    df = pd.read_pickle("Diesel1.pkl")  # Replace with your actual input file
    df = df[df['ds'] >= pd.to_datetime('2015-01-01')].reset_index(drop=True)

    # === 2. CLEAN DATA ===
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    for col in EXOG_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values('ds').reset_index(drop=True)

    # === 3. FEATURE ENGINEERING ===
    df['month_num'] = df['ds'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    for lag in [1, 2, 3, 6]:
        df[f'{TARGET_COL}_lag{lag}'] = df[TARGET_COL].shift(lag)
        for col in EXOG_COLS:
            df[f'{col.replace(" ", "")}_lag{lag}'] = df[col].shift(lag)

    df[f'{TARGET_COL}_roll3'] = df[TARGET_COL].rolling(window=3).mean().shift(1)
    for col in EXOG_COLS:
        df[f'{col.replace(" ", "")}_roll3'] = df[col].rolling(window=3).mean().shift(1)

    # === 4. DEFINE FEATURES & TARGET ===
    features = (
        EXOG_COLS +
        ['month_num', 'month_sin', 'month_cos'] +
        [f'{TARGET_COL}_lag{l}' for l in [1,2,3,6]] +
        [f'{TARGET_COL}_roll3'] +
        [f'{col.replace(" ", "")}_lag1' for col in EXOG_COLS] +
        [f'{col.replace(" ", "")}_lag2' for col in EXOG_COLS] +
        [f'{col.replace(" ", "")}_roll3' for col in EXOG_COLS]
    )

    df_hist = df.dropna(subset=[TARGET_COL] + features).copy()
    X_train = df_hist[features].apply(pd.to_numeric, errors='coerce')
    y_train = pd.to_numeric(df_hist[TARGET_COL], errors='coerce')

    valid_idx = X_train.dropna().index
    X_train = X_train.loc[valid_idx]
    y_train = y_train.loc[valid_idx]

    # === 5. TRUE HOLD-OUT BACKTEST (80% train, 20% test) ===
    df_backtest = df_hist.loc[valid_idx].copy()
    X_all = df_backtest[features]
    y_all = df_backtest[TARGET_COL]

    split_index = int(len(X_all) * 0.8)
    X_train_split, X_test = X_all.iloc[:split_index], X_all.iloc[split_index:]
    y_train_split, y_test = y_all.iloc[:split_index], y_all.iloc[split_index:]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train_split, y_train_split, verbose=False)

    y_pred_test = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    print("\n=== Hold-Out Backtest Error Metrics ===")
    print(f"MAE  (Mean Absolute Error):      {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"MAPE (Mean Absolute % Error):   {mape:.2f}%")

    # === 6. RE-TRAIN ON ALL DATA BEFORE FORECASTING ===
    model.fit(X_train, y_train, verbose=False)

    # === 7. FORECAST MISSING TARGET VALUES ===
    missing_idxs = df.index[df[TARGET_COL].isna()]

    for i in missing_idxs:
        prior = df.loc[:i - 1].copy()
        last = prior.iloc[-1]

        new_feats = {
            'month_num': df.at[i, 'ds'].month,
            'month_sin': np.sin(2 * np.pi * df.at[i, 'ds'].month / 12),
            'month_cos': np.cos(2 * np.pi * df.at[i, 'ds'].month / 12),
            f'{TARGET_COL}_lag1': last[TARGET_COL],
            f'{TARGET_COL}_lag2': last[f'{TARGET_COL}_lag1'],
            f'{TARGET_COL}_lag3': last[f'{TARGET_COL}_lag2'],
            f'{TARGET_COL}_lag6': prior[TARGET_COL].iloc[-6] if len(prior) >= 6 else last[TARGET_COL],
            f'{TARGET_COL}_roll3': pd.to_numeric(prior[TARGET_COL].iloc[-3:], errors='coerce').mean(),
        }

        for col in EXOG_COLS:
            new_feats[col] = df.at[i, col]
            new_feats[f'{col.replace(" ", "")}_lag1'] = last[col]
            new_feats[f'{col.replace(" ", "")}_lag2'] = last[f'{col.replace(" ", "")}_lag1']
            new_feats[f'{col.replace(" ", "")}_roll3'] = pd.to_numeric(prior[col].iloc[-3:], errors='coerce').mean()

        input_df = pd.DataFrame([new_feats])
        input_df = input_df.reindex(columns=features).apply(pd.to_numeric, errors='coerce')
        y_pred = model.predict(input_df)[0]

        df.loc[i, TARGET_COL] = y_pred
        df.loc[i, f'{TARGET_COL}_lag1'] = y_pred
        df.loc[i, f'{TARGET_COL}_lag2'] = last[f'{TARGET_COL}_lag1']
        df.loc[i, f'{TARGET_COL}_lag3'] = last[f'{TARGET_COL}_lag2']
        df.loc[i, f'{TARGET_COL}_lag6'] = prior[TARGET_COL].iloc[-6] if len(prior) >= 6 else y_pred

        roll_vals = pd.concat([prior[TARGET_COL].iloc[-2:], pd.Series([y_pred])])
        df.loc[i, f'{TARGET_COL}_roll3'] = roll_vals.mean() if len(roll_vals) == 3 else y_pred

        for col in EXOG_COLS:
            df.loc[i, f'{col.replace(" ", "")}_lag1'] = last[col]
            df.loc[i, f'{col.replace(" ", "")}_lag2'] = last[f'{col.replace(" ", "")}_lag1']
            df.loc[i, f'{col.replace(" ", "")}_roll3'] = pd.to_numeric(prior[col].iloc[-3:], errors='coerce').mean()

    # === 8. FINAL OUTPUT ===
    df = df[['ds', TARGET_COL]].copy()
    print("\n=== Final df with historical + forecasted values ===")
    print(df.tail(8).to_string(index=False))

    if TARGET_COL not in df_full.columns:
        df_full = pd.merge(df_full, df, on='ds', how='outer')
    else:
        df_full.set_index('ds', inplace=True)
        df.set_index('ds', inplace=True)
        df_full.update(df[[TARGET_COL]])
        df_full.reset_index(inplace=True)

    #=============================#
    #    Optimize Parameters      #
    #=============================#

    """from sklearn.model_selection import GridSearchCV

    # === 5A. HYPERPARAMETER TUNING ===
    # === HYPERPARAMETER SEARCH ONLY ===
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 600, 800],
        'max_depth': [2, 4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.9, 0.8, 0.6],
        'colsample_bytree': [1, 0.8, 0.6]
    }

    grid_model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=grid_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\n=== Best Parameters Found ===")
    print(grid_search.best_params_)"""


    # In[ ]:


    # === 1. SET PARAMETERS ===
    TARGET_COL = 'IND'
    EXOG_COLS = ['WTI', 'Diesel']
    df = pd.read_pickle("Diesel1.pkl")  # Replace with your actual input file
    df = df[df['ds'] >= pd.to_datetime('2015-01-01')].reset_index(drop=True)

    # === 2. CLEAN DATA ===
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    for col in EXOG_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values('ds').reset_index(drop=True)

    # === 3. FEATURE ENGINEERING ===
    df['month_num'] = df['ds'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    for lag in [1, 2, 3, 6]:
        df[f'{TARGET_COL}_lag{lag}'] = df[TARGET_COL].shift(lag)
        for col in EXOG_COLS:
            df[f'{col.replace(" ", "")}_lag{lag}'] = df[col].shift(lag)

    df[f'{TARGET_COL}_roll3'] = df[TARGET_COL].rolling(window=3).mean().shift(1)
    for col in EXOG_COLS:
        df[f'{col.replace(" ", "")}_roll3'] = df[col].rolling(window=3).mean().shift(1)

    # === 4. DEFINE FEATURES & TARGET ===
    features = (
        EXOG_COLS +
        ['month_num', 'month_sin', 'month_cos'] +
        [f'{TARGET_COL}_lag{l}' for l in [1,2,3,6]] +
        [f'{TARGET_COL}_roll3'] +
        [f'{col.replace(" ", "")}_lag1' for col in EXOG_COLS] +
        [f'{col.replace(" ", "")}_lag2' for col in EXOG_COLS] +
        [f'{col.replace(" ", "")}_roll3' for col in EXOG_COLS]
    )

    df_hist = df.dropna(subset=[TARGET_COL] + features).copy()
    X_train = df_hist[features].apply(pd.to_numeric, errors='coerce')
    y_train = pd.to_numeric(df_hist[TARGET_COL], errors='coerce')

    valid_idx = X_train.dropna().index
    X_train = X_train.loc[valid_idx]
    y_train = y_train.loc[valid_idx]

    # === 5. TRUE HOLD-OUT BACKTEST (80% train, 20% test) ===
    df_backtest = df_hist.loc[valid_idx].copy()
    X_all = df_backtest[features]
    y_all = df_backtest[TARGET_COL]

    split_index = int(len(X_all) * 0.8)
    X_train_split, X_test = X_all.iloc[:split_index], X_all.iloc[split_index:]
    y_train_split, y_test = y_all.iloc[:split_index], y_all.iloc[split_index:]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.2,
        subsample=0.6,
        colsample_bytree=0.6,
        random_state=42
    )
    model.fit(X_train_split, y_train_split, verbose=False)

    y_pred_test = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    print("\n=== Hold-Out Backtest Error Metrics ===")
    print(f"MAE  (Mean Absolute Error):      {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"MAPE (Mean Absolute % Error):   {mape:.2f}%")

    # === 6. RE-TRAIN ON ALL DATA BEFORE FORECASTING ===
    model.fit(X_train, y_train, verbose=False)

    # === 7. FORECAST MISSING TARGET VALUES ===
    missing_idxs = df.index[df[TARGET_COL].isna()]

    for i in missing_idxs:
        prior = df.loc[:i - 1].copy()
        last = prior.iloc[-1]

        new_feats = {
            'month_num': df.at[i, 'ds'].month,
            'month_sin': np.sin(2 * np.pi * df.at[i, 'ds'].month / 12),
            'month_cos': np.cos(2 * np.pi * df.at[i, 'ds'].month / 12),
            f'{TARGET_COL}_lag1': last[TARGET_COL],
            f'{TARGET_COL}_lag2': last[f'{TARGET_COL}_lag1'],
            f'{TARGET_COL}_lag3': last[f'{TARGET_COL}_lag2'],
            f'{TARGET_COL}_lag6': prior[TARGET_COL].iloc[-6] if len(prior) >= 6 else last[TARGET_COL],
            f'{TARGET_COL}_roll3': pd.to_numeric(prior[TARGET_COL].iloc[-3:], errors='coerce').mean(),
        }

        for col in EXOG_COLS:
            new_feats[col] = df.at[i, col]
            new_feats[f'{col.replace(" ", "")}_lag1'] = last[col]
            new_feats[f'{col.replace(" ", "")}_lag2'] = last[f'{col.replace(" ", "")}_lag1']
            new_feats[f'{col.replace(" ", "")}_roll3'] = pd.to_numeric(prior[col].iloc[-3:], errors='coerce').mean()

        input_df = pd.DataFrame([new_feats])
        input_df = input_df.reindex(columns=features).apply(pd.to_numeric, errors='coerce')
        y_pred = model.predict(input_df)[0]

        df.loc[i, TARGET_COL] = y_pred
        df.loc[i, f'{TARGET_COL}_lag1'] = y_pred
        df.loc[i, f'{TARGET_COL}_lag2'] = last[f'{TARGET_COL}_lag1']
        df.loc[i, f'{TARGET_COL}_lag3'] = last[f'{TARGET_COL}_lag2']
        df.loc[i, f'{TARGET_COL}_lag6'] = prior[TARGET_COL].iloc[-6] if len(prior) >= 6 else y_pred

        roll_vals = pd.concat([prior[TARGET_COL].iloc[-2:], pd.Series([y_pred])])
        df.loc[i, f'{TARGET_COL}_roll3'] = roll_vals.mean() if len(roll_vals) == 3 else y_pred

        for col in EXOG_COLS:
            df.loc[i, f'{col.replace(" ", "")}_lag1'] = last[col]
            df.loc[i, f'{col.replace(" ", "")}_lag2'] = last[f'{col.replace(" ", "")}_lag1']
            df.loc[i, f'{col.replace(" ", "")}_roll3'] = pd.to_numeric(prior[col].iloc[-3:], errors='coerce').mean()

    # === 8. FINAL OUTPUT ===
    df = df[['ds', TARGET_COL]].copy()
    print("\n=== Final df with historical + forecasted values ===")
    print(df.tail(8).to_string(index=False))

    if TARGET_COL not in df_full.columns:
        df_full = pd.merge(df_full, df, on='ds', how='outer')
    else:
        df_full.set_index('ds', inplace=True)
        df.set_index('ds', inplace=True)
        df_full.update(df[[TARGET_COL]])
        df_full.reset_index(inplace=True)

    #=============================#
    #    Optimize Parameters      #
    #=============================#

    """from sklearn.model_selection import GridSearchCV

    # === 5A. HYPERPARAMETER TUNING ===
    # === HYPERPARAMETER SEARCH ONLY ===
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 600, 800],
        'max_depth': [2, 4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.9, 0.8, 0.6],
        'colsample_bytree': [1, 0.8, 0.6]
    }

    grid_model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=grid_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\n=== Best Parameters Found ===")
    print(grid_search.best_params_)"""


    # In[ ]:


    # === 1. SET PARAMETERS ===
    TARGET_COL = 'RAIL'
    EXOG_COLS = ['WTI', 'Diesel']
    df = pd.read_pickle("Diesel1.pkl")  # Replace with your actual input file
    df = df[df['ds'] >= pd.to_datetime('2015-01-01')].reset_index(drop=True)

    # === 2. CLEAN DATA ===
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    for col in EXOG_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values('ds').reset_index(drop=True)

    # === 3. FEATURE ENGINEERING ===
    df['month_num'] = df['ds'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    for lag in [1, 2, 3, 6]:
        df[f'{TARGET_COL}_lag{lag}'] = df[TARGET_COL].shift(lag)
        for col in EXOG_COLS:
            df[f'{col.replace(" ", "")}_lag{lag}'] = df[col].shift(lag)

    df[f'{TARGET_COL}_roll3'] = df[TARGET_COL].rolling(window=3).mean().shift(1)
    for col in EXOG_COLS:
        df[f'{col.replace(" ", "")}_roll3'] = df[col].rolling(window=3).mean().shift(1)

    # === 4. DEFINE FEATURES & TARGET ===
    features = (
        EXOG_COLS +
        ['month_num', 'month_sin', 'month_cos'] +
        [f'{TARGET_COL}_lag{l}' for l in [1,2,3,6]] +
        [f'{TARGET_COL}_roll3'] +
        [f'{col.replace(" ", "")}_lag1' for col in EXOG_COLS] +
        [f'{col.replace(" ", "")}_lag2' for col in EXOG_COLS] +
        [f'{col.replace(" ", "")}_roll3' for col in EXOG_COLS]
    )

    df_hist = df.dropna(subset=[TARGET_COL] + features).copy()
    X_train = df_hist[features].apply(pd.to_numeric, errors='coerce')
    y_train = pd.to_numeric(df_hist[TARGET_COL], errors='coerce')

    valid_idx = X_train.dropna().index
    X_train = X_train.loc[valid_idx]
    y_train = y_train.loc[valid_idx]

    # === 5. TRUE HOLD-OUT BACKTEST (80% train, 20% test) ===
    df_backtest = df_hist.loc[valid_idx].copy()
    X_all = df_backtest[features]
    y_all = df_backtest[TARGET_COL]

    split_index = int(len(X_all) * 0.8)
    X_train_split, X_test = X_all.iloc[:split_index], X_all.iloc[split_index:]
    y_train_split, y_test = y_all.iloc[:split_index], y_all.iloc[split_index:]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train_split, y_train_split, verbose=False)

    y_pred_test = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    print("\n=== Hold-Out Backtest Error Metrics ===")
    print(f"MAE  (Mean Absolute Error):      {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"MAPE (Mean Absolute % Error):   {mape:.2f}%")

    # === 6. RE-TRAIN ON ALL DATA BEFORE FORECASTING ===
    model.fit(X_train, y_train, verbose=False)

    # === 7. FORECAST MISSING TARGET VALUES ===
    missing_idxs = df.index[df[TARGET_COL].isna()]

    for i in missing_idxs:
        prior = df.loc[:i - 1].copy()
        last = prior.iloc[-1]

        new_feats = {
            'month_num': df.at[i, 'ds'].month,
            'month_sin': np.sin(2 * np.pi * df.at[i, 'ds'].month / 12),
            'month_cos': np.cos(2 * np.pi * df.at[i, 'ds'].month / 12),
            f'{TARGET_COL}_lag1': last[TARGET_COL],
            f'{TARGET_COL}_lag2': last[f'{TARGET_COL}_lag1'],
            f'{TARGET_COL}_lag3': last[f'{TARGET_COL}_lag2'],
            f'{TARGET_COL}_lag6': prior[TARGET_COL].iloc[-6] if len(prior) >= 6 else last[TARGET_COL],
            f'{TARGET_COL}_roll3': pd.to_numeric(prior[TARGET_COL].iloc[-3:], errors='coerce').mean(),
        }

        for col in EXOG_COLS:
            new_feats[col] = df.at[i, col]
            new_feats[f'{col.replace(" ", "")}_lag1'] = last[col]
            new_feats[f'{col.replace(" ", "")}_lag2'] = last[f'{col.replace(" ", "")}_lag1']
            new_feats[f'{col.replace(" ", "")}_roll3'] = pd.to_numeric(prior[col].iloc[-3:], errors='coerce').mean()

        input_df = pd.DataFrame([new_feats])
        input_df = input_df.reindex(columns=features).apply(pd.to_numeric, errors='coerce')
        y_pred = model.predict(input_df)[0]

        df.loc[i, TARGET_COL] = y_pred
        df.loc[i, f'{TARGET_COL}_lag1'] = y_pred
        df.loc[i, f'{TARGET_COL}_lag2'] = last[f'{TARGET_COL}_lag1']
        df.loc[i, f'{TARGET_COL}_lag3'] = last[f'{TARGET_COL}_lag2']
        df.loc[i, f'{TARGET_COL}_lag6'] = prior[TARGET_COL].iloc[-6] if len(prior) >= 6 else y_pred

        roll_vals = pd.concat([prior[TARGET_COL].iloc[-2:], pd.Series([y_pred])])
        df.loc[i, f'{TARGET_COL}_roll3'] = roll_vals.mean() if len(roll_vals) == 3 else y_pred

        for col in EXOG_COLS:
            df.loc[i, f'{col.replace(" ", "")}_lag1'] = last[col]
            df.loc[i, f'{col.replace(" ", "")}_lag2'] = last[f'{col.replace(" ", "")}_lag1']
            df.loc[i, f'{col.replace(" ", "")}_roll3'] = pd.to_numeric(prior[col].iloc[-3:], errors='coerce').mean()

    # === 8. FINAL OUTPUT ===
    df = df[['ds', TARGET_COL]].copy()
    print("\n=== Final df with historical + forecasted values ===")
    print(df.tail(8).to_string(index=False))

    if TARGET_COL not in df_full.columns:
        df_full = pd.merge(df_full, df, on='ds', how='outer')
    else:
        df_full.set_index('ds', inplace=True)
        df.set_index('ds', inplace=True)
        df_full.update(df[[TARGET_COL]])
        df_full.reset_index(inplace=True)

    #=============================#
    #    Optimize Parameters      #
    #=============================#

    """from sklearn.model_selection import GridSearchCV

    # === 5A. HYPERPARAMETER TUNING ===
    # === HYPERPARAMETER SEARCH ONLY ===
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 600, 800],
        'max_depth': [2, 4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.9, 0.8, 0.6],
        'colsample_bytree': [1, 0.8, 0.6]
    }

    grid_model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=grid_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\n=== Best Parameters Found ===")
    print(grid_search.best_params_)"""


    # In[8]:


    df = pd.read_pickle("Diesel1.pkl")
    df = df[['ds', 'SALES']].copy()

    if 'SALES' not in df_full.columns:
        df_full = pd.merge(df_full, df, on='ds', how='outer')
    else:
        df_full.set_index('ds', inplace=True)
        df.set_index('ds', inplace=True)
        df_full.update(df[['SALES']])
        df_full.reset_index(inplace=True)

    df_full = df_full[['ds', 'IND', 'RAIL', 'TRUCKS', 'TONS', 'SALES']].copy()

    # Check if last row (excluding 'ds') is all NaN
    if not df_full.iloc[-1].drop(labels='ds').isna().all():
        next_date = df_full['ds'].max() + relativedelta(months=1)
        new_row = {col: pd.NA for col in df_full.columns}
        new_row['ds'] = next_date
        df_full = pd.concat([df_full, pd.DataFrame([new_row])], ignore_index=True)

    print(df_full.tail())


    # In[ ]:


    # === 1. SET PARAMETERS ===
    TARGET_COL = 'SALES'
    EXOG_COLS = ['IND', 'RAIL', 'TONS', 'TRUCKS']
    df = df_full  # Replace with your actual input file
    df = df[df['ds'] >= pd.to_datetime('2015-01-01')].reset_index(drop=True)

    # === 2. CLEAN DATA ===
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    for col in EXOG_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values('ds').reset_index(drop=True)

    # === 3. FEATURE ENGINEERING ===
    df['month_num'] = df['ds'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    for lag in [1, 2, 3, 6]:
        df[f'{TARGET_COL}_lag{lag}'] = df[TARGET_COL].shift(lag)
        for col in EXOG_COLS:
            df[f'{col.replace(" ", "")}_lag{lag}'] = df[col].shift(lag)

    df[f'{TARGET_COL}_roll3'] = df[TARGET_COL].rolling(window=3).mean().shift(1)
    for col in EXOG_COLS:
        df[f'{col.replace(" ", "")}_roll3'] = df[col].rolling(window=3).mean().shift(1)

    # === 4. DEFINE FEATURES & TARGET ===
    features = (
        EXOG_COLS +
        ['month_num', 'month_sin', 'month_cos'] +
        [f'{TARGET_COL}_lag{l}' for l in [1,2,3,6]] +
        [f'{TARGET_COL}_roll3'] +
        [f'{col.replace(" ", "")}_lag1' for col in EXOG_COLS] +
        [f'{col.replace(" ", "")}_lag2' for col in EXOG_COLS] +
        [f'{col.replace(" ", "")}_roll3' for col in EXOG_COLS]
    )

    df_hist = df.dropna(subset=[TARGET_COL] + features).copy()
    X_train = df_hist[features].apply(pd.to_numeric, errors='coerce')
    y_train = pd.to_numeric(df_hist[TARGET_COL], errors='coerce')

    valid_idx = X_train.dropna().index
    X_train = X_train.loc[valid_idx]
    y_train = y_train.loc[valid_idx]

    # === 5. TRUE HOLD-OUT BACKTEST (80% train, 20% test) ===
    df_backtest = df_hist.loc[valid_idx].copy()
    X_all = df_backtest[features]
    y_all = df_backtest[TARGET_COL]

    split_index = int(len(X_all) * 0.8)
    X_train_split, X_test = X_all.iloc[:split_index], X_all.iloc[split_index:]
    y_train_split, y_test = y_all.iloc[:split_index], y_all.iloc[split_index:]

    model = XGBRegressor(
        n_estimators=800,
        max_depth=2,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train_split, y_train_split, verbose=False)

    y_pred_test = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    print("\n=== Hold-Out Backtest Error Metrics ===")
    print(f"MAE  (Mean Absolute Error):      {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"MAPE (Mean Absolute % Error):   {mape:.2f}%")

    # === 6. RE-TRAIN ON ALL DATA BEFORE FORECASTING ===
    model.fit(X_train, y_train, verbose=False)

    #

    importances = model.feature_importances_
    feature_names = X_train.columns
    sorted_idx = np.argsort(importances)[::-1]

    top_n = 15
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[sorted_idx][:top_n][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in sorted_idx[:top_n]][::-1])
    plt.xlabel("Feature Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.show()

    # === 7. FORECAST MISSING TARGET VALUES ===
    missing_idxs = df.index[df[TARGET_COL].isna()]

    for i in missing_idxs:
        prior = df.loc[:i - 1].copy()
        last = prior.iloc[-1]

        new_feats = {
            'month_num': df.at[i, 'ds'].month,
            'month_sin': np.sin(2 * np.pi * df.at[i, 'ds'].month / 12),
            'month_cos': np.cos(2 * np.pi * df.at[i, 'ds'].month / 12),
            f'{TARGET_COL}_lag1': last[TARGET_COL],
            f'{TARGET_COL}_lag2': last[f'{TARGET_COL}_lag1'],
            f'{TARGET_COL}_lag3': last[f'{TARGET_COL}_lag2'],
            f'{TARGET_COL}_lag6': prior[TARGET_COL].iloc[-6] if len(prior) >= 6 else last[TARGET_COL],
            f'{TARGET_COL}_roll3': pd.to_numeric(prior[TARGET_COL].iloc[-3:], errors='coerce').mean(),
        }

        for col in EXOG_COLS:
            new_feats[col] = df.at[i, col]
            new_feats[f'{col.replace(" ", "")}_lag1'] = last[col]
            new_feats[f'{col.replace(" ", "")}_lag2'] = last[f'{col.replace(" ", "")}_lag1']
            new_feats[f'{col.replace(" ", "")}_roll3'] = pd.to_numeric(prior[col].iloc[-3:], errors='coerce').mean()

        input_df = pd.DataFrame([new_feats])
        input_df = input_df.reindex(columns=features).apply(pd.to_numeric, errors='coerce')
        y_pred = model.predict(input_df)[0]

        df.loc[i, TARGET_COL] = y_pred
        df.loc[i, f'{TARGET_COL}_lag1'] = y_pred
        df.loc[i, f'{TARGET_COL}_lag2'] = last[f'{TARGET_COL}_lag1']
        df.loc[i, f'{TARGET_COL}_lag3'] = last[f'{TARGET_COL}_lag2']
        df.loc[i, f'{TARGET_COL}_lag6'] = prior[TARGET_COL].iloc[-6] if len(prior) >= 6 else y_pred

        roll_vals = pd.concat([prior[TARGET_COL].iloc[-2:], pd.Series([y_pred])])
        df.loc[i, f'{TARGET_COL}_roll3'] = roll_vals.mean() if len(roll_vals) == 3 else y_pred

        for col in EXOG_COLS:
            df.loc[i, f'{col.replace(" ", "")}_lag1'] = last[col]
            df.loc[i, f'{col.replace(" ", "")}_lag2'] = last[f'{col.replace(" ", "")}_lag1']
            df.loc[i, f'{col.replace(" ", "")}_roll3'] = pd.to_numeric(prior[col].iloc[-3:], errors='coerce').mean()

    # === 8. FINAL OUTPUT ===
    df = df[['ds', TARGET_COL]].copy()
    df.to_pickle("DieselD.pkl")
    print("\n=== Final df with historical + forecasted values ===")
    print(df.tail(8).to_string(index=False))


    #=============================#
    #    Optimize Parameters      #
    #=============================#

    """from sklearn.model_selection import GridSearchCV

    # === 5A. HYPERPARAMETER TUNING ===
    # === HYPERPARAMETER SEARCH ONLY ===
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 600, 800],
        'max_depth': [2, 4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.9, 0.8, 0.6],
        'colsample_bytree': [1, 0.8, 0.6]
    }

    grid_model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=grid_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\n=== Best Parameters Found ===")
    print(grid_search.best_params_)"""



if __name__ == "__main__":
    main()
