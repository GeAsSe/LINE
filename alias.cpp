// This file extract the functions of alias sampling method
#include<iostream> 
using namespace std;

int get_norm_prob(int length, double *prob, double *norm_prob) {
    double sum = 0;
    for (int i = 0; i < length; i++) {
        sum += prob[i];
    }
    for (int i = 0; i < length; i++) {
        norm_prob[i] = prob[i] / sum * length;
    }
    return 0;
}

int get_alias_and_prob(int length, double *norm_prob, int *alias, double *prob) {
    int *small = new int[length];
    int *large = new int[length];
    if(small == NULL or large == NULL){
        cout << "Error: memory allocation failed!" << endl;
        return -1;
    }
    int small_index = 0, large_index = 0;
    int cur_small_vertex, cur_large_vertex;
    for (int i = 0; i < length; i++) {
        if (norm_prob[i] < 1) {
            small[small_index] = i;
            small_index++;
        } else {
            large[large_index] = i;
            large_index++;
        }
    }
    while (small_index > 0 && large_index > 0) {
        cur_small_vertex = small[--small_index];
        cur_large_vertex = large[--large_index];

        alias[cur_small_vertex] = cur_large_vertex;
        prob[cur_small_vertex] = norm_prob[cur_small_vertex];

        norm_prob[cur_large_vertex] = norm_prob[cur_large_vertex] + norm_prob[cur_small_vertex] - 1;
        if (norm_prob[cur_large_vertex] < 1) {
            small[small_index] = cur_large_vertex;
            small_index++;
        } else {
            large_index++;
        }
    }
    while (small_index > 0) {
        small_index--;
        prob[small[small_index]] = 1;
    }
    while(large_index > 0) {
        large_index--;
        prob[large[large_index]] = 1;
    }
    free(small);
    free(large);
    return 0;
}