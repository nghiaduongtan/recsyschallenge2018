import numpy as np
import pandas as pd
import random
import pickle
import time
from scipy import sparse
from scipy.sparse.linalg import spsolve
from pyspark import SparkContext

import implicit
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(16)

# Necessary files can be downloaded from OneDrive (see file 'db'). Put these files under the same directory with .py files. 
with open('Y_training_pid_trackid_new_rating_csr.pkl', 'rb') as f:
    Y_training_pid_track_id_rating_sparse_csr = pickle.load(f)

with open('Y_challenge_1_5_10_25_100track_pidnew_trackid_rating_csr.pkl', 'rb') as f:
    Y_challenge_track_pidnew_rating_sparse_csr = pickle.load(f)

Y_training_pid_track_id_rating_sparse_csr = Y_training_pid_track_id_rating_sparse_csr.T
Y_training = sparse.vstack([Y_challenge_track_pidnew_rating_sparse_csr, Y_training_pid_track_id_rating_sparse_csr], 'csr')

W, X = implicit.alternating_least_squares((Y_training*50).astype('double'), 
                                                          factors=400, 
                                                          regularization=0.01, 
                                                          iterations=50, use_gpu=False)


def rec_items(pid):
    pref_vec = Y_training[pid].toarray()
    pref_vec = pref_vec.reshape(-1) + 1
    pref_vec[pref_vec > 1] = 0
    rec_vector = W[pid].dot(X.T)
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
    recommend_vector = pref_vec*rec_vector_scaled
    idx = np.argsort(recommend_vector)[::-1][:500]
    recommendation = pd.DataFrame(idx)
    return recommendation.T

recommend_list = pool.map(rec_items,range(9000))

recommend_list_final = pd.concat(recommend_list)

trackid_trackuri = pd.read_csv('trackid_trackuri_counts.csv')
id_uri_dict = trackid_trackuri.track_uri.to_dict()
recommend_list_final_ax = recommend_list_final.applymap(lambda x: id_uri_dict.get(x) if x in id_uri_dict else x)

with open('Y_challenge_1_5_10_25_100track_pidnew_pid_trackid_rating.pkl', 'rb') as f:
    Y_challenge_1_5_10_25_100track_pidnew_pid_trackid_rating = pickle.load(f)

pid = Y_challenge_1_5_10_25_100track_pidnew_pid_trackid_rating.pid.drop_duplicates(keep='first')
pid.index = range(pid.shape[0])

recommend_list_final_ax.index = range(recommend_list_final_ax.shape[0])
submission_1_5_10_25_100track = pd.concat([pid, recommend_list_final_ax], axis=1, ignore_index=True)

submission_recommend_for_pid_0_track_using_title = pd.read_csv('recommend_pid_0track_using_title.csv', header=None)
submission_recommend_for_challenge_set = pd.concat([submission_1_5_10_25_100track, submission_recommend_for_pid_0_track_using_title], axis=0, ignore_index=True)
submission_recommend_for_challenge_set.to_csv('submission_for_challenge_set.csv', index=False, header=False)
