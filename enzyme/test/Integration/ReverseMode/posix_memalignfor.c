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
#include <stdlib.h>

#include "test_utils.h"

int posix_memalign(void **memptr, size_t alignment, size_t size);

float __enzyme_autodiff(void*, float, int);

float foo(float inp, int n) {
  float* x[10];
  for(int i=0; i<10; i++) {
    posix_memalign((void**)&x[i], 8, 8*(i+1));
    *x[i] = inp;
  }
  float res = *x[10-1];
  return res;
}


int main(int argc, char** argv) {
  float inp = 3.0f;
  float res = __enzyme_autodiff(foo, inp, 32);

  printf("hello! inp=%f, res=%f\n", inp, res);
  APPROX_EQ(res, 1.0f, 1e-10);

  return 0;
}
