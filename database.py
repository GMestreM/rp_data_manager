from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import os
from dotenv import load_dotenv


# Get env variables
load_dotenv()

DB_USER_NAME = os.environ.get('DB_USER_NAME', 'Unable to retrieve DB_USER_NAME') # environ.get
DB_USER_PWD  = os.environ.get('DB_USER_PWD', 'Unable to retrieve DB_USER_PWD')
DB_URL_PATH  = os.environ.get('DB_URL_PATH', 'Unable to retrieve DB_URL_PATH')
DB_URL_PORT  = os.environ.get('DB_URL_PORT', 'Unable to retrieve DB_URL_PORT')

# Create engine
engine = create_engine(f'postgresql://{DB_USER_NAME}:{DB_USER_PWD}@{DB_URL_PATH}:{DB_URL_PORT}/postgres')
engine.connect()

Base = declarative_base()

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()

def query_recent_asset_data():
    """Retrieve the most recent asset price for each date
    
    This function performs an SQL query to retrive
    the most recent asset data for each date: if there
    is more than one value for the day, this function
    obtains those values with a closer 'timestamp' value.

    Returns:
        pd.DataFrame: A dataframe containing one row for
        each id_asset and date, filtered by its maximum timestamp.
    """
    from models import AssetPrices
    from sqlalchemy import func, and_
    import pandas as pd

    sql_subquery = session.query(
        AssetPrices.id_asset,
        AssetPrices.date,
        func.max(AssetPrices.timestamp).label('maxtimestamp')
    ).group_by(AssetPrices.id_asset, AssetPrices.date).subquery('t2')


    sql_query = session.query(AssetPrices).join(
        sql_subquery,
        and_(
            AssetPrices.id_asset == sql_subquery.c.id_asset,
            AssetPrices.date == sql_subquery.c.date,
            AssetPrices.timestamp == sql_subquery.c.maxtimestamp,
        )
    ).order_by(AssetPrices.id_asset, AssetPrices.date)

    df_recent_prices = pd.read_sql(sql_query.statement,session.bind)
    df_recent_prices['date'] = pd.to_datetime(df_recent_prices['date'], format = '%Y-%m-%d')

    return df_recent_prices

def query_recent_model_weights():
    """Retrieve the most recent model weights for each date and model
    
    This function performs an SQL query to retrive
    the most recent model weights for each date and model: if there
    is more than one value for the day, this function
    obtains those values with a closer 'timestamp' value.

    Returns:
        pd.DataFrame: A dataframe containing one row for
        each id_asset, id_model and date, filtered by its maximum timestamp.
    """
    from models import ModelWeights
    from sqlalchemy import func, and_
    import pandas as pd

    sql_subquery = session.query(
        ModelWeights.id_model,
        ModelWeights.id_asset,
        ModelWeights.date,
        func.max(ModelWeights.timestamp).label('maxtimestamp')
    ).group_by(ModelWeights.id_model, ModelWeights.id_asset, ModelWeights.date).subquery('t2')


    sql_query = session.query(ModelWeights).join(
        sql_subquery,
        and_(
            ModelWeights.id_model == sql_subquery.c.id_model,
            ModelWeights.id_asset == sql_subquery.c.id_asset,
            ModelWeights.date == sql_subquery.c.date,
            ModelWeights.timestamp == sql_subquery.c.maxtimestamp,
        )
    ).order_by(ModelWeights.date, ModelWeights.id_asset)

    df_recent_weights = pd.read_sql(sql_query.statement,session.bind)
    df_recent_weights['date'] = pd.to_datetime(df_recent_weights['date'], format = '%Y-%m-%d')

    return df_recent_weights

def init_db_from_scratch():
    """Initialize the database from scratch
    
    This function drops all existing tables in the 
    database, creates new ones and then populates 
    each header table using default values. Then, it
    retrieves asset prices from Investing API; and 
    finally the active Risk Parity model is executed 
    to populate the database with model weights.
    """

    # import all modules here that might define models so that
    # they will be registered properly on the metadata.  Otherwise
    # you will have to import them first before calling init_db()
    from models import AssetHeader, AssetPrices, ModelHeader, ModelWeights
    import pandas as pd
    from datetime import date, datetime
    import investpy
    import time
    import riskfolio as rp


    # Drop all existing tables
    Base.metadata.drop_all(bind=engine)

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Set parameters for API call
    START_DATE   = '01/01/2012'
    END_DATE     = datetime.now().strftime("%d/%m/%Y")

    # Define default data for headers
    dict_default_asset_header = {
        'id':[1,2,3,4,5],
        'isin':[
            'US78462F1030',
            'US4642872349',
            'US4642894798',
            'US78463V1070',
            'US4642877397',
        ],
        'name':[
            'SPDR S&P 500',
            'iShares MSCI Emerging Markets ETF',
            'iShares Core 10+ Year USD Bond ETF (ILTB)',
            'SPDR Gold Shares',
            'iShares U.S. Real Estate ETF',
        ],
        'shortname':[
            'S&P 500',
            'MSCI Emerging Markets',
            'US 10Y Core Bond',
            'Gold',
            'Real Estate',
        ],
        'currency':[
            'USD',
            'USD',
            'USD',
            'USD',
            'USD',
        ],
        'country':[
            'US',
            'US',
            'US',
            'US',
            'US',
        ],
    }

    dict_default_models = {
        'id':[1],
        'flag':[True],
        'assets':['1;2;3;4;5'],
        'window_size':[21*9],
        'constr_dict':[None],
        'rp_model':['Classic'],
        'risk_measure':['MV'],
    }

    # Insert data into header tables
    for i in range(len(dict_default_asset_header['id'])):
        asset_data = AssetHeader(id = dict_default_asset_header['id'][i],
                                 isin = dict_default_asset_header['isin'][i],
                                 name = dict_default_asset_header['name'][i],
                                 shortname = dict_default_asset_header['shortname'][i],
                                 currency = dict_default_asset_header['currency'][i],
                                 country = dict_default_asset_header['country'][i],)
        session.add(asset_data)
    session.commit()

    for i in range(len(dict_default_models['id'])):
        model_data = ModelHeader(id = dict_default_models['id'][i],
                                 flag = dict_default_models['flag'][i],
                                 assets = dict_default_models['assets'][i],
                                 window_size = dict_default_models['window_size'][i],
                                 constr_dict = dict_default_models['constr_dict'][i],
                                 rp_model = dict_default_models['rp_model'][i],
                                 risk_measure = dict_default_models['risk_measure'][i],)
        session.add(model_data)
    session.commit()


    # Insert asset prices from investing API
    # ---------------------------------------

    # First query assets in database
    df_sql_assets = pd.read_sql(session.query(AssetHeader).statement,session.bind)

    # Store historical prices in database
    for idx, row in df_sql_assets.iterrows():
        id_asset   = row['id']
        isin_asset = row['isin']

        # Execute API request
        search_result = investpy.search_quotes(text=isin_asset,
                                               products=['etfs'], 
                                               countries=['United States'], 
                                               n_results=1)

        df_asset_prices = search_result.retrieve_historical_data(from_date=START_DATE, to_date=END_DATE)

        #TO-DO: remove this as it is not needed, just for testing
        asset_price = AssetPrices(id_asset = id_asset,
                                  date = df_asset_prices.index[0],
                                  timestamp = datetime.now(),
                                  open_price = 0,
                                  high_price = 0,
                                  low_price = 0,
                                  close_price = 0,
                                  volume = 1,
                                  change_pct = .1,)
        session.add(asset_price)

        # Pause to avoid having the same timestamp
        time.sleep(1)

        for idx_prices, row_prices in df_asset_prices.iterrows():
            
            # Store data
            asset_price = AssetPrices(id_asset = id_asset,
                                      date = idx_prices,
                                      timestamp = datetime.now(),
                                      open_price = row_prices['Open'],
                                      high_price = row_prices['High'],
                                      low_price = row_prices['Low'],
                                      close_price = row_prices['Close'],
                                      volume = row_prices['Volume'],
                                      change_pct = row_prices['Change Pct'],)
            session.add(asset_price)
        session.commit()

    # Execute and store Risk Parity weights
    # --------------------------------------

    # First read recent historical asset data
    df_recent_asset_prices = query_recent_asset_data()
    
    # Read active model's configuration
    df_sql_models = pd.read_sql(session.query(ModelHeader).statement,session.bind)
    model_row = df_sql_models.head(1)

    # Parse model configuration
    asset_list_model = model_row['assets'].values[0].split(';')
    id_model         = model_row['id']
    window_size      = model_row['window_size'].values[0]
    rp_model         = model_row['rp_model'].values[0]
    risk_measure     = model_row['risk_measure'].values[0]
 
    # Obtain dataframe with asset returns for the RP model
    df_asset_prices_model = pd.DataFrame()
    for asset_id in asset_list_model:
        # Filter
        df_asset_prices_filt = df_recent_asset_prices.loc[df_recent_asset_prices['id_asset'] == int(asset_id),['close_price','date']].copy()
        df_asset_prices_filt.set_index(['date'], inplace=True)
        df_asset_prices_filt.columns = [asset_id]

        # Join
        df_asset_prices_model = pd.concat([df_asset_prices_model, df_asset_prices_filt], axis = 1)

    # TO-DO: use log-returns instead of frac returns    
    df_asset_return_model = df_asset_prices_model.pct_change().fillna(0)

    # Execute Risk Parity model
    df_rp_allocation_backtest = pd.DataFrame()
    for i in range(window_size,len(df_asset_return_model)):
        # TO-DO:
        df_ret_filt = df_asset_return_model.iloc[(i-window_size):(i),:].copy() # It does not use the returns for day D_{i}
        # df_ret_filt = df_asset_return_model.iloc[(i-window_size+1):(i+1),:].copy()
        
        # Initialize portfolio object
        portfolio = rp.Portfolio(returns=df_ret_filt)

        # Estimate statistics relevant for the Risk Parity optimization
        portfolio.assets_stats(method_mu='hist', method_cov='hist')

        # Risk Parity allocation
        rp_weights_aux       = portfolio.rp_optimization(model=rp_model, rm=risk_measure)
        rp_weights_aux       = rp_weights_aux.T
        rp_weights_aux.index = [df_asset_return_model.index[i]]

        df_rp_allocation_backtest = pd.concat([df_rp_allocation_backtest, rp_weights_aux], axis = 0)

    # Store in database
    for idx_weights, row_weights in df_rp_allocation_backtest.iterrows():
        row_weights = row_weights.to_frame().T
        for col_name in row_weights.columns:
            # Store data
            model_weight = ModelWeights(id_model = int(id_model.values[0]),
                                        id_asset = col_name,
                                        date = row_weights.index[0],#idx_weights,
                                        timestamp = datetime.now(),
                                        weight = row_weights[col_name].values[0],)
            session.add(model_weight)
    session.commit()



    # Close current session
    session.close()
    
    
def update_db():
    """Update database with recent data
    
    This function checks the last date with asset data
    in the database and retrieves new prices from Investing's 
    API. If new data is found, the active Risk Parity model is 
    executed to obtain new asset weights for each one of those dates.

    Returns:
        dict: dictionary containing execution information.
    """
    
    from models import AssetHeader, AssetPrices, ModelHeader, ModelWeights
    import pandas as pd
    import numpy as np
    from datetime import date, datetime, timedelta
    import investpy
    import time
    import riskfolio as rp
    
    # Initialize output dictionary
    dict_info_exec = {}
    
    # TO-DO: Query all assets located in the asset header table
    
    # Retrieve asset header
    df_sql_assets = pd.read_sql(session.query(AssetHeader).statement,session.bind)
    
    # Retrieve asset data stored in db
    df_recent_asset_prices = query_recent_asset_data()
    
    # Loop over each asset_id and store the minimum date
    # with data
    unique_asset_id      = df_recent_asset_prices['id_asset'].unique()
    start_date_investing = None
    
    for id_asset in unique_asset_id:
        # Filter dataframe
        df_recent_asset_prices_filt = df_recent_asset_prices.loc[df_recent_asset_prices['id_asset'] == id_asset,:].copy()
        
        # Get maximum date
        max_date_filt =  df_recent_asset_prices_filt['date'].max()
        
        # If the maximum date for current asset 
        # is lower than start_date_investing, use older
        # date
        if start_date_investing is None:
            start_date_investing = max_date_filt
        else:
            if max_date_filt < start_date_investing:
                start_date_investing = max_date_filt
    
    # Next day (?)
    if False: # TO-DO: If False, always repeat last day's asset
        start_date_investing += timedelta(days=1)
    end_date_investing = datetime.now().strftime("%d/%m/%Y")
    # Filter asset header by id_assets  
    df_sql_assets_filt = df_sql_assets.loc[df_sql_assets['id'].isin(unique_asset_id),:].copy()
    print(df_sql_assets_filt)
    
                
    # Query asset prices from Investing API
    list_new_dates = []
    for idx, row in df_sql_assets_filt.iterrows():
        id_asset   = row['id']
        isin_asset = row['isin']
        
        # Execute API request
        search_result = investpy.search_quotes(text=isin_asset,
                                               products=['etfs'], 
                                               countries=['United States'], 
                                               n_results=1)
        try:
            df_asset_prices = search_result.retrieve_historical_data(from_date=start_date_investing.strftime("%d/%m/%Y"),
                                                                     to_date=end_date_investing)
        except:
            df_asset_prices = pd.DataFrame()
            print(f'Unable to retrieve {isin_asset} data between dates {start_date_investing:%d/%m/%Y} and {end_date_investing}')
        
        if False: # TO-DO: Test try-catch
            try:
                df_asset_prices = search_result.retrieve_historical_data(to_date=(start_date_investing + timedelta(days=1)).strftime("%d/%m/%Y"),
                                                                         from_date=start_date_investing.strftime("%d/%m/%Y"))
            except:
                df_asset_prices = pd.DataFrame()
                print(f'Unable to retrieve {isin_asset} data between dates {start_date_investing:%d/%m/%Y} and {end_date_investing}')
            
        # Check if there are new dates
        if len(df_asset_prices) > 0:
            # Get a list of new dates
            list_dates_index = [date for date in df_asset_prices.index]
            
            # Append to new dates
            for date in list_dates_index:
                if date not in list_new_dates:
                    list_new_dates.append(date)
            
            # Store values in database
            for idx_prices, row_prices in df_asset_prices.iterrows():
                # Store data
                asset_price = AssetPrices(id_asset = id_asset,
                                          date = idx_prices,
                                          timestamp = datetime.now(),
                                          open_price = row_prices['Open'],
                                          high_price = row_prices['High'],
                                          low_price = row_prices['Low'],
                                          close_price = row_prices['Close'],
                                          volume = row_prices['Volume'],
                                          change_pct = row_prices['Change Pct'],)
                session.add(asset_price)
            session.commit()
    
    # If data has been stored, execute risk parity for new dates
    if list_new_dates:
        dict_info_exec['new_dates'] = list_new_dates
        
        # First read recent historical asset data
        df_recent_asset_prices = query_recent_asset_data()
    
        # Read active model's configuration
        df_sql_models = pd.read_sql(session.query(ModelHeader).filter(ModelHeader.flag == True).statement,session.bind)
        model_row = df_sql_models.copy()
        
        # Parse model configuration
        asset_list_model = model_row['assets'].values[0].split(';')
        id_model         = model_row['id']
        window_size      = model_row['window_size'].values[0]
        rp_model         = model_row['rp_model'].values[0]
        risk_measure     = model_row['risk_measure'].values[0]
        
        # Obtain dataframe with asset returns for the RP model
        df_asset_prices_model = pd.DataFrame()
        for asset_id in asset_list_model:
            # Filter
            df_asset_prices_filt = df_recent_asset_prices.loc[df_recent_asset_prices['id_asset'] == int(asset_id),['close_price','date']].copy()
            df_asset_prices_filt.set_index(['date'], inplace=True)
            df_asset_prices_filt.columns = [asset_id]

            # Join
            df_asset_prices_model = pd.concat([df_asset_prices_model, df_asset_prices_filt], axis = 1)

        # TO-DO: use log-returns instead of frac returns    
        df_asset_return_model = df_asset_prices_model.pct_change().fillna(0)

        # Filter asset returns to execute model only on new days
        min_start_day = min(list_new_dates)
        
        first_index = int(np.where(df_asset_return_model.index==min_start_day)[0])
        
        # Execute Risk Parity model
        df_rp_allocation_backtest = pd.DataFrame()
        for i in range(first_index,len(df_asset_return_model)):
            # TO-DO:
            # Filter returns (It does not use the returns for day D_{i})
            df_ret_filt = df_asset_return_model.iloc[(i-window_size):(i),:].copy()
            # df_ret_filt = df_asset_return_model.iloc[(i-window_size+1):(i+1),:].copy()
            
            # Initialize portfolio object
            portfolio = rp.Portfolio(returns=df_ret_filt)

            # Estimate statistics relevant for the Risk Parity optimization
            portfolio.assets_stats(method_mu='hist', method_cov='hist')
        
            # Risk Parity allocation
            rp_weights_aux       = portfolio.rp_optimization(model=rp_model, rm=risk_measure)
            rp_weights_aux       = rp_weights_aux.T
            rp_weights_aux.index = [df_asset_return_model.index[i]]

            df_rp_allocation_backtest = pd.concat([df_rp_allocation_backtest, rp_weights_aux], axis = 0)

        # Store in database
        for idx_weights, row_weights in df_rp_allocation_backtest.iterrows():
            row_weights = row_weights.to_frame().T
            for col_name in row_weights.columns:
                # Store data
                model_weight = ModelWeights(id_model = int(id_model.values[0]),
                                            id_asset = col_name,
                                            date = row_weights.index[0],#idx_weights,
                                            timestamp = datetime.now(),
                                            weight = row_weights[col_name].values[0],)
                session.add(model_weight)
        session.commit()
    else:
        # No new dates have been found
        dict_info_exec['new_dates'] = ['None']

    # Close current session
    session.close()
    
    # Return dict with execution information
    return dict_info_exec
