#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "math.h"
#include <ctype.h>


double distance_of_two_points(int d, double *p1, double *p2);
void update_centroids(int d, int k, double **new_centroids, int* clusters_size);
int check_convergence(int d, int k, double epsilon, double **centroids, double **new_centroids);
void assign_closest_cluster(int d, int k, double *data_point, double **centroids, double **new_centroids, int *clusters_size);
int find_closest_cluster(int d, int k, double *data_point, double **centroids);
int* allocate_memory_array_of_size(int k);
double** allocate_memory_array_of_points(int d, int array_size);
void fill_data_list(int d, int n, double **data, PyObject * data_python);
PyObject* save_to_output(int d, int k, double **centroids);
double** runAlg(int d, int k, int n, int max_iter, double epsilon, double **data_points, double **centroids,
            double **new_centroids, int* clusters_size);
static PyObject* fit(PyObject *self, PyObject *args);


static PyObject* fit(PyObject *self, PyObject *args)
{
    int k, max_iter, n, d, *cluster_size;
    double epsilon;
    PyObject * init_centroids_python, * data_points_python, *result;
    double **data_points, **centroids, **new_centroids, **centroids_result;
    if (!PyArg_ParseTuple(args, "iidiOiO", &k, &max_iter, &epsilon, &d, &init_centroids_python, &n, &data_points_python))
        return NULL;
    data_points = allocate_memory_array_of_points(d, n);
    centroids = allocate_memory_array_of_points(d, k);
    new_centroids = allocate_memory_array_of_points(d, k);
    cluster_size = allocate_memory_array_of_size(k);
    fill_data_list(d, n, data_points, data_points_python);
    fill_data_list(d, k, centroids, init_centroids_python);
    save_to_output(d, k, centroids);
    centroids_result = runAlg(d, k, n, max_iter, epsilon, data_points, centroids, new_centroids, cluster_size);
    free(data_points);
    free(cluster_size);
    result = save_to_output(d, k, centroids_result);
    free(centroids);
    free(new_centroids);
    return result;
}


void clear_new_centroids(int d, int k, double **new_centroids, int *clusters_size) {
    int i, j;
    for (i=0 ; i < k ; i++) {
        clusters_size[i] = 0;
        for (j = 0; j < d; j++)
            new_centroids[i][j] = 0;
    }
}

double ** runAlg(int d, int k, int n, int max_iter, double epsilon, double **data_points, double **centroids,
            double **new_centroids, int* clusters_size){
    int iter, i;
    double **temp;
    for (iter=0 ; iter < max_iter ; iter++) {
        for (i=0; i < n ; i++) {
            assign_closest_cluster(d, k, data_points[i], centroids, new_centroids, clusters_size);
        }
        update_centroids(d, k, new_centroids, clusters_size);
        if (check_convergence(d, k, epsilon, centroids, new_centroids) == 1) {
            return new_centroids;
        }
        temp = new_centroids;
        new_centroids = centroids;
        centroids = temp;
        clear_new_centroids(d, k, new_centroids, clusters_size);
    }
    return centroids;
}


void assign_closest_cluster(int d, int k, double *data_point, double **centroids, double **new_centroids, int *clusters_size) {
    int j, i = find_closest_cluster(d, k, data_point, centroids);
    clusters_size[i]++;
    for (j=0; j < d ; j++) {
        new_centroids[i][j] += data_point[j];
    }
}

int find_closest_cluster(int d, int k, double *data_point, double **centroids) {
    int i, closest_cluster = 0;
    double distance, min_distance = distance_of_two_points(d, data_point, centroids[0]);
    for (i=1 ; i < k ; i++) {
        distance = distance_of_two_points(d, data_point, centroids[i]);
        if (distance < min_distance) {
            min_distance = distance;
            closest_cluster = i;
        }
    }
    return closest_cluster;
}

void update_centroids(int d, int k, double **new_centroids, int* clusters_size) {
    int i, j;
    for (i=0; i < k ; i++) {
        for (j=0; j < d ; j++) {
            new_centroids[i][j] /= clusters_size[i];
        }
    }
}

int check_convergence(int d, int k, double epsilon, double **centroids, double **new_centroids) {
    int i;
    for (i=0; i < k ; i++) {
        double distance = distance_of_two_points(d, centroids[i], new_centroids[i]);
        if (sqrt(distance) >= epsilon) {
            return 0;
        }
    }
    return 1;
}

double distance_of_two_points(int d, double *p1, double *p2) {
    int i;
    double result = 0;
    for (i=0; i < d ; i++) {
        result += pow(p1[i] - p2[i], 2);
    }
    return result;
}

int* allocate_memory_array_of_size(int k) {
    int* a = calloc(k, sizeof(int));
    assert(a != NULL);
    return a;
}

double** allocate_memory_array_of_points(int d, int array_size) {
    double *p;
    double **a;
    int i;
    p = calloc(d * array_size, sizeof(double));
    a = calloc(array_size, sizeof(double *));
    for(i=0 ; i < array_size ; i++ )
        a[i] = p+ i * d;
    assert(a != NULL);
    assert(p != NULL);
    return a;
}


void fill_data_list(int d, int n, double **data, PyObject * data_python) {
    int i, j;
    PyObject* inner_list, * py_float;
    for (i = 0 ; i < n ; i++) {
        inner_list = PyList_GetItem(data_python, i);
        for (j = 0; j < d; j++) {
            py_float = PyList_GetItem(inner_list, j);
            data[i][j] = PyFloat_AsDouble(py_float);
        }
    }
}

PyObject* save_to_output(int d, int k, double **centroids) {
    int i, j;
    PyObject *val, *inner_list, *result;
    result = PyList_New(k);
    for (i = 0 ; i < k ; i++) {
        inner_list = PyList_New(d);
        for (j = 0; j < d; j++) {
            val = PyFloat_FromDouble(centroids[i][j]);
            PyList_SetItem(inner_list, j, val);
        }
        PyList_SetItem(result, i, inner_list);
    }
    return result;
}

static PyMethodDef kmeansMethods[] = {
        {"fit",
        (PyCFunction) fit,
        METH_VARARGS,
        PyDoc_STR("kmeans alg")},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp",
        NULL,
        -1,
        kmeansMethods
};

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
