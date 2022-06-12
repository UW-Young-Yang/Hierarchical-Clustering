# import pandas as pd
import numpy as np
import csv
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

def load_data(filepath):
    with open(filepath, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)
    # df = pd.read_csv(filepath)
    # return df.to_dict(orient='records')

def calc_features(row):
    return np.array([row['Attack'], row['Sp. Atk'], row['Speed'], row['Defense'], row['Sp. Def'], row['HP']], dtype='int64')


def hac(features):
    distance_matrix = np.zeros((len(features), len(features)))
    cluster_dict = dict()
    res = np.zeros((len(features)-1, 4))

    for row_idx in range(len(features)):
        cluster_dict[row_idx] = [row_idx]
        for col_idx in range(len(features)):
            distance_matrix[row_idx, col_idx] = np.linalg.norm(features[row_idx]-features[col_idx])
    
    for i in range(len(features)-1):
        minimum = None
        for cluster1_idx in cluster_dict:
            for cluster2_idx in cluster_dict:
                if cluster2_idx <= cluster1_idx:
                    continue
                distance = 0
                for pokemon1 in cluster_dict[cluster1_idx]:
                    for pokemon2 in cluster_dict[cluster2_idx]:
                        if distance_matrix[pokemon1, pokemon2] > distance:
                            distance = distance_matrix[pokemon1, pokemon2]
                if minimum == None or minimum[2] > distance:
                    minimum = (cluster1_idx, cluster2_idx, distance)
        cluster1_idx = minimum[0]
        cluster2_idx = minimum[1]
        distance = minimum[2]
        cluster_dict[len(features)+i] = cluster_dict[cluster1_idx] + cluster_dict[cluster2_idx]
        cluster_dict.pop(cluster1_idx)
        cluster_dict.pop(cluster2_idx)
        res[i, 0] = cluster1_idx
        res[i, 1] = cluster2_idx
        res[i, 2] = distance
        res[i, 3] = len(cluster_dict[len(features)+i])
    return res

def imshow_hac(Z):
    hierarchy.dendrogram(Z)
    plt.show()