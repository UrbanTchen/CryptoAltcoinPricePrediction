from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


class AltcoinPredictor:
    
    def __init__(self, df, cnt_valid=14):
        self.df = df
        self.cnt_valid = cnt_valid
        self.train = None
        self.valid = None
    
    
    def split_train_valid(self):
        """
        Splits the dataset into a Training and Validation Set
        The Validation Set consist of the last "cnt_valid"
        days of the Time Series "df" passed
        """
        
        self.train = self.df.iloc[:-self.cnt_valid]
        self.valid = self.df.iloc[-self.cnt_valid:]
        # Define whether Altcoin's Price increased-or-equal or decreased
        # for the last day of the Prediction Period vs. last day of Training Set
        self.y_usd_up_or_equal = self.valid['priceUsd'][-1] >= self.train['priceUsd'][-1]
        self.y_btc_up_or_equal = self.valid['priceBtc'][-1] >= self.train['priceBtc'][-1]
    
    
    def fit_model(self):
        """
        Use Vector Autoregression, pass Training Set & fit the model
        """
        
        model = VAR(endog = self.train)
        self.model_fit = model.fit()
    
        
    def forecast(self):
        """
        Forecast the next x days, where x is the number of days
        in the Validation Set
        """
        
        # Get Prediction for all Features in the DataFrame as a numpy array
        prediction = self.model_fit.forecast(self.model_fit.endog, steps=len(self.valid))
        # Convert n-dim numpy array to a Prediction DataFrame
        self.pred_df = pd.DataFrame(prediction,
                                    index=self.valid.index,
                                    columns=self.train.columns)
        # Prediction whether Altcoin's Price increased-or-equal or decreased
        self.yhat_usd_up_or_equal = self.pred_df['priceUsd'][-1] >= self.train['priceUsd'][-1]
        self.yhat_btc_up_or_equal = self.pred_df['priceBtc'][-1] >= self.train['priceBtc'][-1]
    
    
    def fit_predict(self):
        """
        Does fit and forecast function in one step
        """
        
        self.fit_model()
        self.forecast()
    
    
    def get_pred_summary(self):
        """
        Create a DataFrame showing the relative and absolute
        Error of the Altcoin's USD & BTC Price Prediction vs. True Price
        
        - saves the DataFrame as instance variable
        (used to be print an overview of plot it)
        """
        
        # Get Prices from Prediction DataFrame
        pred_prices = self.pred_df[['priceBtc', 'priceUsd']]
        # Get Prices from Validation DataFrame
        valid_prices = self.valid[['priceBtc', 'priceUsd']]
        # Show Altcoin's BTC Price Prediction & True Price as DataFrame (to plot)
        self.btc_pred = pd.concat([pred_prices['priceBtc'], valid_prices['priceBtc']], keys=['prediction', 'truePrice'], axis=1)
        # Show Altcoin's USD Price Prediction & True Price as DataFrame (to plot)
        self.usd_pred = pd.concat([pred_prices['priceUsd'], valid_prices['priceUsd']], keys=['prediction', 'truePrice'], axis=1)
        
        # Concat both USD & BTC Price Prediction with True Prices
        # to calculate the relative and absolute Prediction Error
        price_diff = pd.concat([pred_prices, valid_prices],
                            axis=1, keys=['pred', 'valid'])
        # Add column: Relative Price Prediction vs. True Price Error
        price_diff['rel_btc_diff'] = price_diff['pred', 'priceBtc'] / price_diff['valid', 'priceBtc'] - 1
        price_diff['rel_usd_diff'] = price_diff['pred', 'priceUsd'] / price_diff['valid', 'priceUsd'] - 1
        # Add column: Absolute Price Prediction vs. True Price Error
        price_diff['abs_btc_diff'] = price_diff['pred', 'priceBtc'] - price_diff['valid', 'priceBtc']
        price_diff['abs_usd_diff'] = price_diff['pred', 'priceUsd'] - price_diff['valid', 'priceUsd']
        
        # Set the Price Difference DataFrame as an object variable
        self.price_diff = price_diff
        
    
    def calculate_performance(self):
        """
        Calculate Performance for the Error Metrics (nRMSE, MAPE
        and Price increased-or-equal or decreased) for Altcoin's
        USD & BTC Price Prediction
        
        - saves Error Metrics as several instance variables
        - variable "error_dict" stores all Error Metrics as a dict
        """
        
        # Get Prices from Prediction & True Price (Validation) DataFrame
        y_true_usd, y_pred_usd = self.valid['priceUsd'], self.pred_df['priceUsd']
        y_true_btc, y_pred_btc = self.valid['priceBtc'], self.pred_df['priceBtc']
        # Calculate normalized RMSE between Prediction & Validation
        rmse_usd = sqrt(mean_squared_error(y_true_usd, y_pred_usd))
        nrmse_usd = rmse_usd / self.valid['priceUsd'].mean()
        rmse_btc = sqrt(mean_squared_error(y_true_btc, y_pred_btc))
        nrmse_btc = rmse_btc / self.valid['priceBtc'].mean()
        # Set normalized RMSE as instance variable
        self.nrmse_usd = round(nrmse_usd, 3)
        self.nrmse_btc = round(nrmse_btc, 3)
        
        # Calculate MAPE between Prediction & Validation
        mape_usd = np.mean(np.abs((y_true_usd - y_pred_usd) / y_true_usd)) * 100
        mape_btc = np.mean(np.abs((y_true_btc - y_pred_btc) / y_true_btc)) * 100
        # Set MAPE as instance variable
        self.mape_usd = round(mape_usd, 3)
        self.mape_btc = round(mape_btc, 3)
        
        # True/False whether Altcoin's Price increased-or-equal or decreased is correct
        self.usd_movement_correct = self.yhat_usd_up_or_equal == self.y_usd_up_or_equal
        self.btc_movement_correct = self.yhat_btc_up_or_equal == self.y_btc_up_or_equal
        
        # Save all Error Metrics as dict
        # (will be accessed when doing multiple period predictions)
        self.error_dict = {
                            'nRMSE USD': self.nrmse_usd,
                            'MAPE USD': self.mape_usd,
                            'nRMSE BTC': self.nrmse_btc,
                            'MAPE BTC': self.mape_btc,
                            'USD Movement correct': self.usd_movement_correct,
                            'BTC Movement correct': self.btc_movement_correct
                        }
    
    def print_performance(self):
        """
        Prints the Performance as a quick Summary when needed
        Shows # of Training & Validation Units, nRMSE & MAPE
        for the Altcoin's USD & BTC Prediction
        """
        
        print('Train: %s, Pred: %s\n-----' % (len(self.train), len(self.pred_df)))
        print('nRMSE (USD): %s' % self.nrmse_usd)
        print('MAPE (USD): %s' % self.mape_usd)
        print('nRMSE (BTC): %s' % self.nrmse_btc)
        print('MAPE (BTC): %s' % self.mape_btc)
    
    
    def plot_predictions(self, save_name=False):
        """
        Plots the Altcoin's USD & BTC Price Prediction vs. True Price
        which was calculated in function "get_pred_summary()"
        """
        
        # Initialize 2 Subplots
        fig, ax = plt.subplots(2, 1)
        
        # Plot Altcoin's BTC Price Prediction along with True Price
        ax[0].plot(self.btc_pred)
        ax[0].title.set_text('Price Prediction in BTC value')
        ax[0].tick_params('x', labelrotation=45)
        # Plot Altcoin's USD Price Prediction along with True Price
        ax[1].plot(self.usd_pred)
        ax[1].title.set_text('Price Prediction in USD value')
        ax[1].tick_params('x', labelrotation=45)
        
        plt.tight_layout()
        # Save Plot as png if "save_name" is defined
        if save_name:
            plt.savefig('Price Prediction vs True Prices in BTC and USD (%s).png' % save_name)
        plt.show()
    
    
    def split_fit_predict_error(self):
        """
        Executes all steps (split dataset, fit+predict, calculate error, get summary)
        """
        
        self.split_train_valid()
        self.fit_model()
        self.forecast()
        self.calculate_performance()
        self.get_pred_summary()