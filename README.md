# Crypto Altcoin Price Prediction with Multivariate Time Series

_"Cryptocurrency prices are mainly driven by emotions."_

What we can do:

- Past research has shown that Twitter data can be used to predict market movements of securities. Some research papers also tried to predict crypto prices using sentiment analysis.
- Santiment collects and aggregates data about blockchain companies from various data sources. Santiment data is still very new (available since 2018).

For more information on Santiment data, visit `sanpy`: https://github.com/santiment/sanpy

Goal: Twitter Sentiment vs. Blockchain Activity Data
---
![Poster Crypto Altcoin Price Prediction](images/thesis_poster.png)

The goal of my thesis was to find out if we can use blockchain activity data (provided by Santiment) to make better price predictions than with Tweet’s sentiment data.

The Notebook `CryptoAltcoinPricePrediction.ipynb` is the main project file which contains the complete analysis for my BSc Computer Science (Major Data Science) Thesis in a nutshell.

Method
---
I handpicked 3 blockchain projects to make the price predictions on (market cap data as of mid February 2020):

- Enjin Coin (`#enjin`), Number 59, 105 million USD market cap
- OmiseGo (`#omisego`), Number 45, 152 million USD market cap
- Chainlink (`#chainlink`), Number 12, 1.5 billion USD market cap

I retrieved Twitter using their hashtags & Santiment data for the period 1st Jan 2019 - 18th Feb 2020. There were several steps required to have a ready-to-use time series to be fed into a Machine Learning algorithm. This process can be found in the Jupyter Notebook.

A VAR (Vector Autoregression) with default maxlags (12 * (nobs/100.)**(1./4)) was chosen as the algorithm to make the performance comparison. The focus of my research was on feature engineering.

Dataset
---
The data that I prepared has been saved to a folder `data`, so I can use them in the Notebook.

The following files contain all Tweets with sentiment scores & magnitudes (via Google Natural Language API) in JSON format:

- `omisego_all_sentiment_1y_v2.txt`
- `enjin_all_sentiment_1y_v2.txt`
- `chainlink_all_sentiment_1y_v2.txt`

The following files contain Santiment blockchain activity data for 1 year (only available with Santiment Premium) in a Python DataFrame. In the Notebook, however, I retrieve them with `sanpy` using an API key:

- `san_omisego_20190101_20200118.csv`
- `san_enjin_20190101_20200118.csv`
- `san_chainlink_20190101_20200118.csv`

Python Classes/Files
---
I wrote 4 Python classes to make my work more efficient:

`TwitterAltcoinData.py` is created to:
- set parameters for the Twitter API calls
- retrieve the attributes needed for my work
- and to write the extracted Tweets into a file locally

I saved the downloaded Twitter data in txt files. So there are untouched in the Jupyter Notebook. However, the code can be used to retrieve Tweets of your choice.

`SantimentAltcoinData.py` is used to:
- set project (altcoin), start & end date for which Santiment's blockchain activity data should be retrieved
- retrieve single time series for a list of Santiment metrics passed as parameter, and concatenate them into a features dataframe
- and to concatenate the altcoin price to the features dataframe

`AltcoinPredictor.py` is the most important class & time-saver of my work and is used to:
- initialize an object with a ready to use multivariate time series (e.g. the dataframe built with `SantimentAltcoinData`)
- split the dataframe into a training set & validation set based on a "cnt_valid" parameter (number of units to be used for validation set)
- fit + predict a VAR (Vector Autoregression) model with default maxlags (12 * (nobs/100.)**(1./4))
- save prediction output and error metrics (MAPE & nRMSE) in attributes
- print error metrics of the prediction after fit + predict
- plot prediction vs. real price

`util.py` is a utility class that contains several functions, in order to avoid repeating codes