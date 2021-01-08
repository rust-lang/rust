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

double __enzyme_autodiff(void*, ...);

double alldiv(double* a, int N, double start) {
  for (int i=0; i<N; i++) {
    if (a[i] > start)
      start = a[i];
  }
  return start;
}

double alldiv2(double* a, int N) {
  double start = 0.0f;
  for (int i=0; i<N; i++) {
    if (a[i] > start)
      start = a[i];
  }
  return start;
}
int main(int argc, char** argv) {

  int N = 5;
  double a[N];
  for(int i=0; i<N; i++) {
    a[i] = i+1;
  }

  a[3] = 102.;

  double d_a[N];
  for(int i=0; i<N; i++)
    d_a[i] = 0.0f;
  
  double res = alldiv(a, N, 3.14);
  //omp(*a, N);
  printf("ran res=%e\n", res);
  double dstart = __enzyme_autodiff((void*)alldiv, a, d_a, N, 3.14);

  for(int i=0; i<N; i++) {
    printf("a[%d]=%f  d_a[%d]=%e  cor=%e\n", i, a[i], i, d_a[i], - res / a[i]);
  }
  fflush(0);
  for(int i=0; i<N; i++) {
    printf("i=%d\n", i);
    APPROX_EQ(d_a[i], (i == 3) ? 1 : 0, 1e-10);
  }
  APPROX_EQ(dstart, 0, 1e-10);

  for(int i=0; i<N; i++) {
    a[i] = i+1;
  }
  a[4] = 102.;


  for(int i=0; i<N; i++)
    d_a[i] = 0.0f;
  __enzyme_autodiff((void*)alldiv2, a, d_a, N);
  for(int i=0; i<N; i++) {
    printf("a[%d]=%f  d_a[%d]=%e  cor=%e\n", i, a[i], i, d_a[i], - res / a[i]);
  }
  fflush(0);
  for(int i=0; i<N; i++) {
    printf("i=%d\n", i);
    APPROX_EQ(d_a[i], (i == 4) ? 1 : 0, 1e-10);
  }

  for(int i=0; i<N; i++) {
    a[i] = i+1;
  }


  for(int i=0; i<N; i++)
    d_a[i] = 0.0f;
  dstart = __enzyme_autodiff((void*)alldiv, a, d_a, N, 102.);
  for(int i=0; i<N; i++) {
    printf("a[%d]=%f  d_a[%d]=%e  cor=%e\n", i, a[i], i, d_a[i], - res / a[i]);
  }
  fflush(0);
  for(int i=0; i<N; i++) {
    printf("i=%d\n", i);
    APPROX_EQ(d_a[i], 0, 1e-10);
  }
  APPROX_EQ(dstart, 1, 1e-10);
  return 0;
}
