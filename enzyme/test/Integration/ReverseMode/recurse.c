//   note not doing O0 below as to ensure we get tbaa
// RUN: %clang -std=c11 %O0TBAA %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
//   note not doing O0 below as to ensure we get tbaa
// RUN: %clang -std=c11 %O0TBAA %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

#define __builtin_autodiff __enzyme_autodiff
double __enzyme_autodiff(void*, ...);

double recurse_max_helper(float* a, float* b, int N) {
  if (N <= 0) {
    return *a + *b;
  }
  return recurse_max_helper(a,b,N-1) + recurse_max_helper(a,b,N-2);
}
void recurse_max(float* a, float* b, float* ret, int N) {
  *ret = recurse_max_helper(a,b,N);
}

int main(int argc, char** argv) {
  float a = 2.0;
  float b = 3.0;

  float da = 0;
  float db = 0;

  float ret = 0;
  float dret = 2.0;

  //recurse_max(&a, &b, &ret, 20);

  int N = 20;
  int dN = 0;

  __builtin_autodiff(recurse_max, &a, &da, &b, &db, &ret, &dret, 20);

  APPROX_EQ(da, 17711.0*2, 1e-10);
  APPROX_EQ(db, 17711.0*2, 1e-10);

  printf("hello! %f, res2 %f, da: %f, db: %f\n", ret, ret, da,db);
  return 0;
}
