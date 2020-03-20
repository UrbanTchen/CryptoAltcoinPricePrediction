from searchtweets import ResultStream, gen_rule_payload, load_credentials, collect_results
import jsonpickle
import datetime
import pandas as pd
#import tweepy (used when using the Standard Twitter API)


class TwitterAltcoinData:
    def __init__(self, cred_file, yaml_key):
        """
        Initialize an object with loading the credentials
        using a credentials file and yaml key
        """
        self.premium_search_args = load_credentials(cred_file,
                                       yaml_key=yaml_key,
                                       env_overwrite=False)
    
    
    def get_tweet_attributes(self, tw):
        """
        Get only the necessary fields of a tweet
        
        - returns a Tweet with the needed attributes as a dict
        """
        
        tw_dict = {}
        
        tw_dict['created_at'] = tw['created_at']
        tw_dict['lang'] = tw['lang']
        tw_dict['text'] = tw['text']
        tw_dict['entities'] = tw['entities']
        tw_dict['favorite_count'] = tw['favorite_count']
        tw_dict['retweet_count'] = tw['retweet_count']
        tw_dict['user_followers_cnt'] = tw['user']['followers_count']
        tw_dict['user_following_cnt'] = tw['user']['friends_count']
        tw_dict['user_id'] = tw['user']['id_str']
        
        return tw_dict
    
    
    def premium_set_search_params(self, search_query, from_date, to_date,
                                    no_retweets=True, results_per_call=500):
        """
        Sets the Search Query and maximum Tweets
        to be retrieved to save Quota
        """
        
        # Set a static Language Filter for English Tweets
        lang_filter = ' lang:en'
        if no_retweets:
            rt_filter = ' -is:retweet'
            # Adds an ignore Retweets tag to the (Altcoin) Query
            self.query = search_query + lang_filter + rt_filter
        else:
            # This Query includes all Tweets, also Retweets
            self.query = search_query + lang_filter
        # Sets the Rule for the Query to be executed (time frame & # of Results)
        self.rule = gen_rule_payload(self.query, results_per_call=results_per_call,
                                    from_date=from_date, to_date=to_date)
    
    
    def premium_download_save_tweets(self, file_name, max_results=100):
        """
        Downloads all Tweets since from_date for a Query
        and saves them into txt File (in append mode)
        """
        
        tweets = collect_results(self.rule,
                        max_results=max_results,
                        result_stream_args=self.premium_search_args)
        # save all tweets into specified file_name
        with open(file_name, 'a+') as f:
            for i, tweet in enumerate(tweets):
                if i % 100 == 0:
                    print('write tweet %s to %s' % (i, file_name))
                tw = self.get_tweet_attributes(tweet)
                f.write(jsonpickle.encode(tw, unpicklable=False) + '\n')