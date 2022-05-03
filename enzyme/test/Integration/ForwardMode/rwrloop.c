// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -fno-unroll-loops -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
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

double __enzyme_fwddiff(void*, ...);

double alldiv(double* __restrict__ a, int *__restrict__ N) {
  double sum = 0;
  for(int i=0; i<10; i++) {
    for(int j=0; j<*N; j++) {
        sum += a[10*i+j] * a[10*i+j];
        a[10*i+j] = 0;
    }
  }
  *N = 7;
  return sum;
}

int main(int argc, char** argv) {

  int N = 10;
  double a[N][N];
  for(int i=0; i<N; i++) {
      for(int j=0; j<N; j++) {
      a[i][j] = i*100+j;
    }
  }

  double d_a[N][N];
  for(int i=0; i<N; i++) {
      for(int j=0; j<N; j++) {
      d_a[i][j] = 1;
    }
  }
  
  double d_start = __enzyme_fwddiff((void*)alldiv, (double*)a, (double*)d_a, &N);
  N = 10;

  printf("d_start=%f\n", d_start);
  APPROX_EQ(d_start, 90900.0, 1e-10 );

  return 0;
}
