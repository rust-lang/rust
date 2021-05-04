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

double __enzyme_autodiff(void*, ...);

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
  memset((void*)d_a, 0, sizeof(d_a));
  
  double dstart = __enzyme_autodiff((void*)alldiv, (double*)a, (double*)d_a, &N);
  N = 10;

  for(int i=0; i<N; i++) {
      for(int j=0; j<N; j++) {
          printf("d_a[%d][%d]=%f\n", i, j, d_a[i][j]);
      }
  }
  fflush(0);
  for(int i=0; i<N; i++) {
      for(int j=0; j<N; j++) {
        APPROX_EQ(d_a[i][j], 2. * (i*100+j), 1e-8 );
    }
  }
  return 0;
}
