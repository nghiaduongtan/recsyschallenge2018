import numpy as np
import pandas as pd
import random
import pickle
import time
from scipy import sparse
import json
import re
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(16)

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

with open('challenge_set.json') as f:
    challenge_set = json.load(f)

challenge_set_df        = pd.DataFrame(challenge_set['playlists'])
challenge_set_df_       = challenge_set_df[['name','num_samples','pid']]
challenge_set_df_0track = challenge_set_df_[challenge_set_df_.num_samples == 0]

challenge_set_df_0track_normalized = challenge_set_df_0track.copy()

challenge_set_df_0track_normalized['name_normalized'] = challenge_set_df_0track.name.map(normalize_name)

challenge_0_tracks_title = challenge_set_df_0track_normalized.name_normalized.values
pid_0_track = challenge_set_df_0track_normalized.pid.values

trackid_trackuri = pd.read_csv('trackid_trackuri_counts.csv')
track_id_uri_dict = trackid_trackuri.track_uri.to_dict()
uri_id_dict = dict((k,v) for (v,k) in track_id_uri_dict.items())

# All playlist in million playlist dataset have the title in common with 10,000 playlist challenge 
with open('title_in_training.pkl','rb') as f:
    title_in_training = pickle.load(f)

def recommend(pid_index):
    list_recommend = pd.DataFrame()
    pid = pid_0_track[pid_index]
    pid_name = challenge_0_tracks_title[pid_index]
    if pid_name == 'universal stereo':
        # recommend most popular track for playlist 
        list_recommend[pid_index] = range(500)
        return list_recommend.T
    title_track_df = sum_title_challenge.loc[challenge_title_tracks[pid_name]].track_uri.value_counts()
    track_recommend = title_track_df.index
    track_recommend = list(track_recommend) + list(range(500))
    track_recommend_s = pd.Series(track_recommend)
    track_recommend_s.drop_duplicates(keep='first',inplace=True)
    list_recommend[pid_index] = track_recommend_s.values[:500]
    return list_recommend.T

list_recommendation = pool.map(recommend,range(1000))
recommendation      = pd.concat(list_recommendation)
pid_n = challenge_set_df_0track_normalized.pid
pid_n.index = range(pid_n.shape[0])

recommendation_ax = recommendation.applymap(lambda x: track_id_uri_dict.get(x) if x in track_id_uri_dict else x)
submission = pd.concat([pid_n,recommendation_ax],axis=1,ignore_index=True)
submission.to_csv('data_new_rating_5th/recommend_pid_0track_using_title.csv',index=False,header=False)
