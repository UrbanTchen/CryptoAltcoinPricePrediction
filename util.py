import os
import json
import jsonpickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt
from google.cloud import language

import tweepy
from searchtweets import ResultStream, gen_rule_payload, load_credentials, collect_results
import san


def read_txt_json(file_name):
    """
    Reads a txt File
    Used to load the saved Tweets
    with Sentiment Scores & Magnitudes
    
    - returns a list of dicts for every Tweet
    """
    
    tweets = []
    for line in open(file_name, 'r'):
        tweets.append(json.loads(line))
    
    return tweets


def tweets_to_dataframe(tweets):
    """
    Converts a list of all Tweet dicts into a DataFrame
    where each row represents a Tweet
    
    - returns a DataFrame
    """
    
    tweet_list = []
    
    # a tw represents a single Tweet
    for tw in tweets:
        # Get the Number of Hashtags for the Tweet
        cnt_hashtags = len(tw['entities']['hashtags'])
        # Get the Number of Cashtags for the Tweet
        cnt_symbols = len(tw['entities']['symbols'])
        # Add a list of Attributes for the Tweet into a list
        tweet_list.append([tw['created_at'],
                            tw['lang'],
                            tw['sentiment_score'],
                            tw['sentiment_magnitude'],
                            tw['favorite_count'],
                            tw['retweet_count'],
                            tw['user_followers_cnt'],
                            cnt_hashtags,
                            cnt_symbols
                        ])
    # Create a list of column names for the DataFrame
    cols = ['created_at', 'lang', 'sentiment_score', 'sentiment_magnitude',
            'favorite_count', 'retweet_count', 'user_followers_cnt',
            'cnt_hashtags', 'cnt_symbols']
    # Create a DataFrame with the Tweets list & use the defined column names
    df = pd.DataFrame(tweet_list, columns=cols)
    # Add a date field as a column (make it time-zone = None, so they can be joined later)
    df['date'] = pd.to_datetime(df['created_at']).dt.floor('d').dt.tz_localize(None)
    return df


def sentiment_score_range(x):
    """
    Using the insights about the frequency distribution
    of the sentiment scores, I defined the following ranges.
    
    - returns the sentiment score range of the input
    """
    
    if x <= -0.3:
        return 'sco_less_eq_neg_0.3'
    elif -0.3 < x <= 0:
        return 'sco_greater_neg_0.3'
    elif 0 < x <= 0.2:
        return 'sco_greater_0'
    elif 0.2 < x <= 0.5:
        return 'sco_greater_0.2'
    elif 0.5 < x <= 0.7:
        return 'sco_greater_0.5'
    elif 0.7 < x:
        return 'sco_greater_0.7'

def sentiment_magnitude_range(x):
    """
    Using the insights about the frequency distribution
    of the sentiment magnitude, I defined the following ranges.
    
    - returns the sentiment score range of the input
    """
    
    if x <= 0.3:
        return 'mag_less_eq_0.3'
    elif 0.3 < x <= 0.6:
        return 'mag_greater_0.3'
    elif 0.6 < x <= 0.8:
        return 'mag_greater_0.6'
    elif 0.8 < x:
        return 'mag_greater_0.8'


def add_adjusted_sentiment_ranges(df):
    """
    Calculates how much weight a user's Tweet should have
    (log increased by the Number of Followers)
    
    All users with <= 100 followers, have weight 1.
    User with 1'000 has weight 2
    User with 10'000 has weight 3 etc.
    
    The sentiment scores & magnitude of the Tweet
    will be multiplied with the user's "Influence Multiplier"
    
    - returns the input DataFrame with the newly added columns
    """
    
    # Compute the user's weight to be used as an "influence multipier"
    df['followers_multiplier_base100'] = df.user_followers_cnt.apply(lambda x: np.log10(max(x/10, 10)))
    # Add 2 columns for the weight adjusted sentiment score & magnitude
    df['adj100_sentiment_score'] = df['sentiment_score'] * df['followers_multiplier_base100']
    df['adj100_sentiment_magnitude'] = df['sentiment_magnitude'] * df['followers_multiplier_base100']
    # Use the weight adjusted sentiments to put them into score & magnitude range
    df['adj100_sentiment_score_range'] = df['adj100_sentiment_score'].apply(lambda x: sentiment_score_range(x))
    df['adj100_sentiment_magnitude_range'] = df['adj100_sentiment_magnitude'].apply(lambda x: sentiment_magnitude_range(x))
    # Do the same, but with the original sentiment score & magnitude of the Tweet
    df['sentiment_score_range'] = df['sentiment_score'].apply(lambda x: sentiment_score_range(x))
    df['sentiment_magnitude_range'] = df['sentiment_magnitude'].apply(lambda x: sentiment_magnitude_range(x))
    
    return df


def predict_multi_periods(df, dates, predictor, verbose=False):
    """
    For each of 'dates', it creates an AltcoinPredictor with the above
    dataframe till the according end date,
    
    splits the dataframe into a training & validation set, where
    the 7 days of the dataframe is the validation set,
    
    then, fit + predicts using VAR of order 1 and saves all
    error metrics and price predictions in a dictionary
    
    - returns 2 dicts (error metrics & price predictions for each period)
    """
    
    # dict to store Error Metrics of all Predictions for each Period
    pred_dict = {
                'nRMSE USD': [],
                'MAPE USD': [],
                'nRMSE BTC': [],
                'MAPE BTC': [],
                'USD Movement correct': [],
                'BTC Movement correct': []
            }
    # dict to store Altcoin's USD & BTC Price Predictions for each Period
    pred_plot_dict = {
                'USD pred': [],
                'BTC pred': []
            }
            
    # For every Prediction Period (d is the end date of a Dataset)
    for d in dates:
        # Create a predictor object (usually AltcoinPredictor) till date "d"
        alt_clf = predictor(df.loc[:d].fillna(0), 7)
        # Execute all steps (split dataset, fit+predict, calculate error, get summary)
        alt_clf.split_fit_predict_error()
        
        for x in pred_dict:
            ## Append Error Metrics from predictor to dict
            #
            # "error_dict" of predictor stores Error Metrics in a dict
            # with the same keys as "pred_dict"
            pred_dict[x].append(alt_clf.error_dict[x])
        # Append Altcoin's USD & BTC Price Prediction vs. True Price to dict
        pred_plot_dict['USD pred'].append(alt_clf.usd_pred)
        pred_plot_dict['BTC pred'].append(alt_clf.btc_pred)
    # Print Average of Error Metrics if verbose
    if verbose:
        for (err_metric, values) in pred_dict.items(): 
            print(err_metric + ': ' + str(round(np.mean(values), 4)))
    # return 2 dicts (Error Metrics & Price Predictions)
    return pred_dict, pred_plot_dict


def plot_features(df):
    """
    Plots each Feature as a Line Chart in a Subplot
    """
    
    # Drop Prices to get all price-predicting Features only
    df = df.drop(['priceUsd', 'priceBtc'], axis=1, errors='ignore')
    # Number of Features
    cnt_features = len(df.columns)
    # Create Figure with the number of Subplots
    fig, axes = plt.subplots(nrows=round(cnt_features/2), ncols=2, dpi=120, figsize=(10,6))
    # Plot each Feature in a separate Subplot
    for i, ax in enumerate(axes.flatten()):
        # Break, to prevent index out of range
        if i == cnt_features:
            break
        # Get the Feature's time series to plot
        data = df[df.columns[i]]
        ax.plot(data, color='red', linewidth=1)
        # Set column value as Title and some Formatting
        ax.set_title(df.columns[i])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.show()


def plot_predictions(data, dates, save_name=False):
    """
    Take a list of DataFrames containing the Altcoin's
    USD Prediction & Real Price,
    then plots them one by one in a Subplot.
    """
    
    # Create Figure with the number of Subplots
    fig, axes = plt.subplots(nrows=round(len(data)/2) , ncols=2, dpi=120, figsize=(6,8))
    # Plot every Price Prediction in a separate Subplot
    for i, ax in enumerate(axes.flatten()):
        # Break, to prevent index out of range
        if i == len(data):
            break
        ax.plot(data[i], linewidth=1)
        # Set Date as Title and some Formatting
        ax.set_title(dates[i])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.tick_params('x', labelrotation=25)
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
        ax.legend(data[i].columns, loc='best', prop={'size': 6})
    plt.tight_layout()
    # Save Plot as png if "save_name" is defined
    if save_name:
        plt.savefig('Price Prediction vs True Prices in USD (%s).png' % save_name)
    plt.show()
    
    
def concat_price_to_project_metrics(metrics_df, project,
                                    price_units=['priceBtc', 'priceUsd'],
                                    interval="1d"):
    """
    Concatenates the Price to the passed Santiment Metrics DataFrame for the Altcoin
    
    - returns DataFrame with Price concatenated
    """
    # Set start & end date for the Price Retrieval
    start_date = metrics_df.index.min().strftime('%Y-%m-%d')
    end_date = metrics_df.index.max().strftime('%Y-%m-%d')
    # Get Price Data for the Period for which we have Metrics
    price_df = san.get(
                    'prices/' + project,
                    from_date=start_date,
                    to_date=end_date,
                    interval=interval
            )
    # Make both DataFrames tz-naive, so we can join them
    price_df.index = price_df.index.map(lambda x: x.tz_localize(None))
    metrics_df.index = metrics_df.index.map(lambda x: x.tz_localize(None))
    # return a DataFrame consisting of Metrics with concatenated Price
    return pd.concat([metrics_df, price_df[price_units]], axis=1)


def grangers_causation_matrix(data, variables, test='ssr_chi2test', maxlag=12, verbose=False):    
    """
    Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors.
    
    The values in the table are the p-Values.
    p-Values lesser than the significance level (0.05), implies that
    the Null Hypothesis that the X does not cause Y can be rejected.
    
    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    
    Function taken from machinelearningplus.com.
    """
    
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [str(var) + '_x' for var in variables]
    df.index = [str(var) + '_y' for var in variables]
    return df


def adfuller_test(series, signif=0.05, name='', verbose=True):
    """
    Perform ADFuller to test for Stationarity
    of given series and print report
    
    Function taken from machinelearningplus.com and rewritten by me.
    """
    
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)
    
    if p_value <= signif:
        hypothesis_action = 'Rejecting H0.'
        stationary = True
    else:
        hypothesis_action = 'Weak evidence to reject H0.'
        stationary = False
    
    # Print Output
    if verbose:
        # Print Summary
        print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
        print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
        print(f' Significance Level    = {signif}')
        print(f' Test Statistic        = {output["test_statistic"]}')
        print(f' No. Lags Chosen       = {output["n_lags"]}')
        
        for key,val in r[4].items():
            print(f' Critical value {adjust(key)} = {round(val, 3)}')
        # Hypothesis Action & Is Series Stationary?    
        print(f" => P-Value = {p_value}. {hypothesis_action}")
        if p_value <= signif:
            print(f" => Series is Stationary.")
        else:
            print(f" => Series is Non-Stationary.")
    # return values for the p-value table
    return name, p_value, stationary


# To obtain Sentiment Score & Magnitude of a Tweet.
# Not needed for Notebook: I saved all Tweets with Sentiment Information in txt files
def gc_sentiment(text, client):
    """
    Passes a given text through Google's Natural Language API
    
    - returns Sentiment Score & Magnitude as a tuple
    """
    try:
        document = language.types.Document(
                content=text,
                type=language.enums.Document.Type.PLAIN_TEXT)
        annotations = client.analyze_sentiment(document=document)
        score = annotations.document_sentiment.score
        magnitude = annotations.document_sentiment.magnitude
        return round(score, 3), round(magnitude, 3)
    except:
        return None, None


# Not needed for Notebook: I saved all Tweets with Sentiment Information in txt files
def add_sentiment_save_tweet(tweets, file_name, client, limit=10):
    """
    Takes a list of Tweets, then passes the text to gc_sentiment()
    to obtain Sentiment Score & Magnitude
    
    - The results are added back to the Tweet as a dict
    - List of dicts (each Tweet) is saved to a txt file
    """
    # a+ : create file if it doesn't exist and open it in append mode
    with open(file_name, 'a+') as f:
        for i, tw in enumerate(tweets):
            # in order to not use too much quota by accident
            if i >= limit > None:
                return
            # Obtain Sentiment Score & Magnitude of a Tweet
            score, magnitude = gc_sentiment(tw['text'], client)
            # Add Score & Magnitude to dict
            tw['sentiment_score'] = score
            tw['sentiment_magnitude'] = magnitude
            # Write each updated dict (Tweet) into txt file
            f.write(json.dumps(tw) + '\n')
            if i % 10 == 0:
                print('write tweet %s to file' % i)
        f.close()