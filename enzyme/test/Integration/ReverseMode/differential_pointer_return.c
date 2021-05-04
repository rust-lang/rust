// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

#define __builtin_autodiff __enzyme_autodiff
double __enzyme_autodiff(void*, ...);

double f_read(double* x) {
  double product = (*x) * (*x);
  return product;
}

double* g_write(double* x, double product) {
  *x = (*x) * product;
  return x;
}

double h_read(double* x) {
  return *x;
}

double readwriteread_helper(double* x) {
  double product = f_read(x);
  x = g_write(x, product);
  double ret = h_read(x);
  return ret; 
}

void readwriteread(double*__restrict x, double*__restrict ret) {
  *ret = readwriteread_helper(x);
}

int main(int argc, char** argv) {
  double ret = 0;
  double dret = 1.0;
  double* x = (double*) malloc(sizeof(double));
  double* dx = (double*) malloc(sizeof(double));
  *x = 2.0;
  *dx = 0.0;

  __builtin_autodiff(readwriteread, x, dx, &ret, &dret);

  
  printf("dx is %f ret is %f\n", *dx, ret);
  APPROX_EQ(*dx, 3*2.0*2.0, 1e-10);
  return 0;
}
