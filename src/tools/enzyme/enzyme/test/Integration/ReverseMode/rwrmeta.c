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

__attribute__((noinline))
double loadSq(double**__restrict__ x) {
    double* __restrict__ one = *x;
    double  two = *one;
    return two * two;
}
double alldiv(double** __restrict__ x) {
  double* __restrict__ one = *x;
  double res = loadSq(x);
  one[0] = 3.14159265;
  return res;
}

int main(int argc, char** argv) {
  double  x = 3.0;
  double dx = 0.0;
  double*  xx = &x;
  double* dxx = &dx;
  double**  xxx = &xx;
  double** dxxx = &dxx;
  
  __enzyme_autodiff((void*)alldiv, xxx, dxxx);
  APPROX_EQ(dx, 6.0, 1e-6);
  return 0;
}
