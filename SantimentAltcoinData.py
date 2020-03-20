import san
import pandas as pd
import settings
san.ApiConfig.api_key = settings.API_KEY


class SantimentAltcoinData:
    
    def __init__(self, project, start_date, end_date):
        self.project = project
        self.start_date = start_date
        self.end_date = end_date
    
                                       
    def get_metric(self, metric, interval="1d"):
        """
        Retrieve a Santiment Metric
        
        - returns a DataFrame with 1 column
        """
        data = san.get(
            metric + '/' + self.project,
            from_date=self.start_date,
            to_date=self.end_date,
            interval=interval
        )
        data = data.rename(columns={data.columns[0]: metric})
        return data
    
    
    def get_metrics(self, metrics, interval="1d"):
        """
        Retrieves a list of Santiment Metrics
        and concatenates them
        
        - sets a DataFrame with all Features
        """
        
        df = pd.DataFrame()
        
        for m in metrics:
            # Get a Santiment Metric
            metric_df = self.get_metric(m, interval)
            # Add it to the DataFrame
            df = pd.concat([df, metric_df], axis=1)
        # Save DataFrame as an instance variable
        self.df = df
    
    
    def concat_price_to_df(self,
                            price_units=['priceBtc', 'priceUsd'],
                            interval="1d"):
        """
        Retrieves the Price Time Series
        and concatenates them to the Metrics DataFrame
        """
                            
        # Set start & end date for the Price to be retrieved
        start_date = self.df.index.min().strftime('%Y-%m-%d')
        end_date = self.df.index.max().strftime('%Y-%m-%d')
        # Get Price Data for the period for which we have metrics
        price_df = self.get_metric('prices')
        # Make both DataFrames tz-naive, so we can join them
        price_df.index = price_df.index.map(lambda x: x.tz_localize(None))
        self.df.index = self.df.index.map(lambda x: x.tz_localize(None))
        
        # Set Metrics with Price joined as new DataFrame
        self.df = pd.concat([self.df, price_df[price_units]], axis=1).dropna()