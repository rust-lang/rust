
// RUN: clang -std=c11 -O1 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -S | lli - 


// RUN: clang -std=c11 -O0 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_inline=1 -S | lli - 
// RUN: clang -std=c11 -O1 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_inline=1 -S | lli - 

// RUN: clang -std=c11 -O3 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_inline=1 -S | lli - 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

#define __builtin_autodiff __enzyme_autodiff

double __enzyme_autodiff(void*, ...);

// size of array
float* unsorted_array_init(int N) {
  float* arr = (float*) malloc(sizeof(float) * N);
  for (int i = 0; i < N; i++) {
    arr[i] = 1.0*(i%2);
  }
  return arr;
}

void insertsort_sum (float*__restrict array, int N, float*__restrict ret) {
  float sum = 0;

  for (int i = 1; i < N; i++) {
    int j = i;
    while (j > 0 && array[j-1] < array[j]) {
      float tmp = array[j];
      array[j] = array[j-1];
      array[j-1] = tmp;
      j -= 1;
    }
  }

  for (int i = 0; i < N/2; i++) {
    //printf("Val: %f\n", array[i]);
    sum += array[i];
  }

  *ret = sum;
}


int main(int argc, char** argv) {
  float ret = 0;
  float dret = 1.0;

  int N = 10;
  int dN = 0;
  float* array = unsorted_array_init(N);
  float* d_array = (float*) malloc(sizeof(float)*N);
  for (int i = 0; i < N; i++) {
    d_array[i] = 0.0;
  }

  printf("Array before sorting:\n");
  for (int i = 0; i < N; i++) {
    printf("%d:%f\n", i, array[i]); 
  }

  __builtin_autodiff(insertsort_sum, array, d_array, N, &ret, &dret);

  printf("The total sum is %f\n", ret);

  printf("Array after sorting:\n");
  for (int i = 0; i < N; i++) {
    printf("%d:%f\n", i, array[i]); 
  }

  for (int i = 0; i < N; i++) {
    printf("Diffe for index %d is %f\n", i, d_array[i]);
    if (i%2 == 0) {
      APPROX_EQ(d_array[i], 0.0, 1e-10);
    } else {
      APPROX_EQ(d_array[i],1.0,1e-10);
    }
  }
  return 0;
}
