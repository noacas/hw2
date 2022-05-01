import sys
import numpy as np
import pandas as pd
#import mykmeanssp

DEFAULT_MAX_ITER = 300


def invalid_input():
    print('Invalid Input!')
    exit()


def parse_args():
    k, max_iter, eps, file_name_1, file_name_2 = None, None, None, None, None
    if len(sys.argv) == 6:
        k, max_iter, eps, file_name_1, file_name_2 = sys.argv[1:]
    elif len(sys.argv) == 5:
        k, eps, input_file, output_file = sys.argv[1:]
        max_iter = DEFAULT_MAX_ITER
    else:
        invalid_input()
    if (not k.isnumeric()) or (not max_iter.isnumeric()):
        invalid_input()
    try:
        eps = float(eps)
    except:
        invalid_input()
    k = int(k)
    max_iter = int(max_iter)
    if (k <= 1) or (max_iter <= 0):
        invalid_input()
    if not (file_name_1.endswith('.csv') or file_name_1.endswith('.txt')) or \
            not (file_name_2.endswith('.csv') or file_name_2.endswith('.txt')):
        invalid_input()
    return k, max_iter, eps, file_name_1, file_name_2


def kmeans_pp(k, data_points):
    np.random.seed(0)
    indices = data_points[0].to_numpy()
    #m = data_points.sample(random_state=np.random.RandomState())
    add_to_m_index = np.random.choice(indices)
    m = data_points.loc[data_points[0] == add_to_m_index]
    for i in range(k-1):
        d = data_points.apply(min_distance, axis=1, args=[m])
        sum_d = d.sum()
        p = d.apply(lambda x: x / sum_d)
        #add_to_m = data_points.sample(random_state=np.random.RandomState(), weights=p)
        add_to_m_index = np.random.choice(indices, p=p)
        add_to_m = data_points.loc[data_points[0] == add_to_m_index]
        m = pd.concat([m, add_to_m], ignore_index=True)
    return m.pop(0), m


def min_distance(data_point, m):
    diff = m.apply(lambda x: x - data_point, axis=1)
    diff.pop(0)
    distance = diff.apply(np.linalg.norm, axis=1)
    return distance.min()


def print_result(indices, result):
    print(','.join([str(int(i)) for i in indices]))
    result.to_csv(sys.stdout, header=False, index=False,
                  float_format='%.4f')


def parse_input_files(file_name_1, file_name_2, k):
    files = [pd.read_csv(file_name_1, header=None),
             pd.read_csv(file_name_2, header=None)]
    data_points = pd.merge(files[0], files[1], how='inner', on=0)
    if k >= data_points.shape[0]:
        invalid_input()
    return data_points


def main():
    k, max_iter, eps, file_name_1, file_name_2 = parse_args()
    data_points = parse_input_files(file_name_1, file_name_2, k)
    indices, init_centroids = kmeans_pp(k, data_points)
    data_points.pop(0)
    print(','.join([str(int(i)) for i in indices]))
    #result = mykmeanssp.fit(k, max_iter, init_centroids, len(data_points), data_points)
    #print_result(indices, result)


if __name__ == '__main__':
    try:
        main()
    except:
        print('An Error Has Occurred')