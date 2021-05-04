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


float __enzyme_autodiff(void*, float, int);

float foo(float inp, int n) {
  float* x = 0;
  for(int i=0; i<n; i++) {
    x = (float*)realloc(x, (i+1)*sizeof(float));
    if (i == 0 ){
        *x = inp;
    } else {
        x[i] = x[i-1] + inp;
    }
  }
  float res = x[n-1];
  free(x);
  return res;
}


int main(int argc, char** argv) {
  float inp = 3.0f;
  float res = __enzyme_autodiff(foo, inp, 32);

  printf("hello! inp=%f, res=%f\n", inp, res);
  APPROX_EQ(res, 32.0f, 1e-10);

  return 0;
}
