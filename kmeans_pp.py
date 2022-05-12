import math
import sys
import numpy as np
import pandas as pd
import mykmeanssp

DEFAULT_MAX_ITER = 300


class InvalidInput(Exception):
    pass


def parse_args():
    k, max_iter, eps, file_name_1, file_name_2 = None, None, None, None, None
    if len(sys.argv) == 6:
        k, max_iter, eps, file_name_1, file_name_2 = sys.argv[1:]
        if not max_iter.isnumeric():
            raise InvalidInput
        max_iter = int(max_iter)
    elif len(sys.argv) == 5:
        k, eps, file_name_1, file_name_2 = sys.argv[1:]
        max_iter = DEFAULT_MAX_ITER
    else:
        raise InvalidInput
    if not k.isnumeric():
        raise InvalidInput
    k = int(k)
    try:
        eps = float(eps)
    except:
        raise InvalidInput
    if (k <= 1) or (max_iter <= 0):
        raise InvalidInput
    if not (file_name_1.endswith('.csv') or file_name_1.endswith('.txt')) or \
            not (file_name_2.endswith('.csv') or file_name_2.endswith('.txt')):
        raise InvalidInput
    return k, max_iter, eps, file_name_1, file_name_2


def kmeans_pp(k, data_points):
    np.random.seed(0)
    indices = data_points[0].to_numpy()
    data_points = data_points.to_numpy()
    data = {indices[i]: data_points[i][1:] for i in range(indices.size)}
    choice = np.random.choice(indices)
    m = {choice: data[choice]}
    for i in range(k-1):
        d = [calc_min_dis(point, m) for point in data.values()]
        sum_d = sum(d)
        p = [x / sum_d for x in d]
        choice = np.random.choice(indices, p=p)
        m[choice] = data[choice]
    return m.keys(), [line.tolist() for line in m.values()]


def calc_min_dis(point, m):
    cur_min = math.inf
    for other_point in m.values():
        sum = 0
        for v, v2 in zip(point, other_point):
            sum += math.pow(v-v2, 2)
        if sum < cur_min:
            cur_min = sum
    return cur_min


def min_distance(data_point, m):
    diff = m.apply(lambda x: x - data_point, axis=1)
    diff.pop(0)
    power = diff.apply(lambda x: pow(x, 2), axis=1)
    distance = power.sum(axis=1)
    return distance.min()


def print_result(result):
    for centroid in result:
        print(','.join(str(float("%.4f" % data)) for data in centroid))


def parse_input_files(file_name_1, file_name_2, k):
    files = [pd.read_csv(file_name_1, header=None),
             pd.read_csv(file_name_2, header=None)]
    data_points = pd.merge(files[0], files[1], how='inner', on=0)
    if k >= data_points.shape[0]:
        raise InvalidInput
    data_points = data_points.sort_values(by=[0])
    return data_points


def main():
    k, max_iter, eps, file_name_1, file_name_2 = parse_args()
    data_points = parse_input_files(file_name_1, file_name_2, k)
    indices, init_centroids = kmeans_pp(k, data_points)
    print(','.join([str(int(i)) for i in indices]))
    data_points.pop(0)
    data_points = data_points.values.tolist()
    result = mykmeanssp.fit(k, max_iter, eps, len(data_points[0]), init_centroids, len(data_points), data_points)
    print_result(result)


if __name__ == '__main__':
    try:
        main()
    except InvalidInput:
        print('Invalid Input!')
    except:
        print('An Error Has Occurred')