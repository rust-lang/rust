// RUN: %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

extern double __enzyme_fwddiff(void*, double*, double*, int);

double sumtil(double* vec, int size) {
  double ret = 0.0;
  for (int i = 0; i < size; i++) {
    ret += vec[i] * vec[i];
    if (ret > 30) break;
    ret += vec[i] * vec[i];
    vec[i] = 0;
  }
  return ret;
}

int main() {
    double vec[] = {1, 2., 3., 4., 5.};
    double d_vec[] = {2., 0., 1., 9., 8.};
    double d_ret = __enzyme_fwddiff((void*)sumtil, vec, d_vec, 5);

    printf("d_ret=%f\n", d_ret);
    APPROX_EQ(d_ret, 92.0, 1e-7);

    double args[] = {0, 0, 0, 9, 8};
    for(int i=0; i<5; i++) {
      printf("i=%d d_vec=%f ans=%f\n", i, d_vec[i], args[i]);
      APPROX_EQ(d_vec[i], args[i], 1e-7);
    }
    printf("done\n");
}