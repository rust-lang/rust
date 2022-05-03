// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

void __enzyme_autodiff(void*, ...);

void alldiv(double* a, int N) {
  for (int i=0; i<=N; i++) {
    a[i] *= a[i];
  }
}

int main(int argc, char** argv) {

  double* a1 = malloc(sizeof(double)); 
  double* a10 = malloc(10*sizeof(double)); 
  alldiv((double*)0, -1);
  alldiv((double*)0, -10);
  alldiv(a1, 0);
  alldiv(a10, 9);
  //omp(*a, N);
  printf("ran orig\n");
  fflush(0);

  double* d_a1 = malloc(sizeof(double)); 
  double* d_a10 = malloc(10*sizeof(double)); 

  __enzyme_autodiff((void*)alldiv, (double*)0, (double*)0, -1);
  __enzyme_autodiff((void*)alldiv, (double*)0, (double*)0, -10);

  a1[0] = 3.14;
  d_a1[0] = 1.0;
  
  __enzyme_autodiff((void*)alldiv, (double*)a1, (double*)d_a1, 0);

  printf("a1[0]=%f  d_a1[0]=%e\n", a1[0], d_a1[0]);
  fflush(0);
  APPROX_EQ(a1[0], 3.14*3.14, 1e-10);
  APPROX_EQ(d_a1[0], 6.28, 1e-10);

  for (int i=0; i<10; i++) {
      a10[i] = 1+i;
      d_a10[i] = 1;
  }

  __enzyme_autodiff((void*)alldiv, (double*)a10, (double*)d_a10, 9);

  for(int i=0; i<10; i++) {
    printf("a10[%d]=%f  d_a10[%d]=%e\n", i, a10[i], i, d_a10[i]);
  }
  fflush(0);
  for(int i=0; i<10; i++) {
    APPROX_EQ(d_a10[i], 2.0 * (1+i), 1e-10);
  }
  return 0;
}
