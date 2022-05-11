#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <ctype.h>
#include <errno.h>

double const EPSILON = 0.001;
int const DEFAULT_MAX_ITER = 200;


double distance_of_two_points(int d, double *p1, double *p2);
void update_centroids(int d, int k, double **new_centroids, int* clusters_size);
int check_convergence(int d, int k, double **centroids, double **new_centroids);
void assign_closest_cluster(int d, int k, double *data_point, double **centroids, double **new_centroids, int *clusters_size);
int find_closest_cluster(int d, int k, double *data_point, double **centroids);
int invalid_input();
void my_error();

void initialize_centroids(int d, int k, double **data_points, double **centroids) {
    int i, j;
    for (i=0 ; i < k ; i++)
        for (j=0 ; j < d ; j++)
            centroids[i][j] = data_points[i][j];
}

void clear_new_centroids(int d, int k, double **new_centroids, int *cluster_size) {
    int i, j;
    for (i=0 ; i < k ; i++) {
        cluster_size[i] = 0;
        for (j = 0; j < d; j++)
            new_centroids[i][j] = 0;
    }
}

void runAlg(int d, int k, int n, int max_iter, double **data_points, double **centroids,
            double **new_centroids, int* clusters_size){
    double **temp;
    int iter, i;
    initialize_centroids(d, k, data_points, centroids);
    for (iter=0 ; iter < max_iter ; iter++) {
        for (i=0; i < n ; i++) {
            assign_closest_cluster(d, k, data_points[i], centroids, new_centroids, clusters_size);
        }
        update_centroids(d, k, new_centroids, clusters_size);
        if (check_convergence(d, k, centroids, new_centroids) == 1) {
            break;
        }
        temp = new_centroids;
        new_centroids = centroids;
        centroids = temp;
        clear_new_centroids(d, k, new_centroids, clusters_size);
    }
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

int check_convergence(int d, int k, double **centroids, double **new_centroids) {
    int i;
    for (i=0; i < k ; i++) {
        double distance = distance_of_two_points(d, centroids[i], new_centroids[i]);
        if (sqrt(distance) >= EPSILON) {
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

void find_d_n(int *result, char* input) {
    FILE *fileptr;
    int n = 1, d = 0, ch=0;;
    fileptr = fopen(input, "r");
    if (fileptr == NULL) {
        my_error();
    }
    while ((ch = fgetc(fileptr)) != '\n') {
        if (ch == ',')
            d++;
    }
    while(!feof(fileptr))
    {
        ch = fgetc(fileptr);
        if(ch == '\n')
            n++;
    }
    fclose(fileptr);
    result[0] = d + 1;
    result[1] = n;
}


void get_data_points(int d, int n, double **data_points, char *input) {
    FILE *f;
    int result, i, j;
    char str[32], *p;
    double data;
    f = fopen(input, "r");
    for (i = 0 ; i < n ; i++) {
        for (j = 0; j < d; j++) {
            result = fscanf(f, "%31[^,\n]", str);
            if (result == 0)
                my_error();
            data = strtod(str, &p);
            data_points[i][j] = data;
            result = fscanf(f, "%c*", str);
        }
    }
    fclose(f);
}

void save_to_output(int d, int k, double **centroids, char *output) {
    FILE *f;
    int i, j;
    f = fopen(output, "w");
    for (i = 0 ; i < k ; i++) {
        for (j = 0; j < d-1; j++) {
            fprintf(f, "%.4f,", centroids[i][j]);
        }
        fprintf(f, "%.4f\n", centroids[i][d-1]);
    }
    fclose(f);
}


void run_logic(int k, int max_iter, char* input, char* output) {
    int result[2], d, n, *cluster_size;
    double **data_points, **centroids, **new_centroids;
    find_d_n(result, input);
    d = result[0];
    n = result[1];
    data_points = allocate_memory_array_of_points(result[0], result[1]);
    centroids = allocate_memory_array_of_points(d, k);
    new_centroids = allocate_memory_array_of_points(d, k);
    cluster_size = allocate_memory_array_of_size(k);
    get_data_points(d, n, data_points, input);
    runAlg(d, k, n, max_iter, data_points, centroids, new_centroids, cluster_size);
    free(data_points);
    free(new_centroids);
    free(cluster_size);
    save_to_output(d, k, centroids, output);
    free(centroids);
}

int invalid_input(){
    printf("Invalid Input!");
    return 1;
}

void my_error(){
    printf("An Error Has Occurred");
    exit(1);
}


int main(int argc, char *argv[]) {
    char *p;
    int k, max_iter = DEFAULT_MAX_ITER;;
    if ((argc != 5) && (argc != 4))
        return invalid_input();

    k = strtol(argv[1], &p, 10);
    if (*p != '\0' || errno != 0 || k <= 1)
        return invalid_input();

    if (argc == 5) {
        max_iter =  strtol(argv[2], &p, 10);
        if (*p != '\0' || errno != 0 || max_iter < 1)
            return invalid_input();
    }
    run_logic(k, max_iter, argv[argc-2], argv[argc-1]);
    return 0;
}

