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

extern void __enzyme_autodiff(void*, double*, double*, int);
/*double max(double x, double y) {
    return (x > y) ? x : y;
}*/

double sumtil(double* vec, int size) {
  double ret = 0.0;
  double tmp[4];
  for (int i = 0; i < size; i++) {
    for(int j=0; j<4; j++) {
      tmp[j] = i * j * j;
    }
    for(int j=0; j<4; j++) {
      ret += tmp[j] * vec[i];
    }
  }
  return ret;
}

int main() {
    double vec[] = {1, 2., 3., 4., 5.};
    double d_vec[] = {0., 0., 0., 0., 0.};
    __enzyme_autodiff(sumtil, vec, d_vec, 5);

    for(int i=0; i<5; i++) {
      printf("d_reduce_max(%i)=%f\n", i, d_vec[i]);
    }
    fflush(0);
    for(int i=0; i<5; i++) {
      printf("i=%d d_vec=%f ans=%f\n", i, d_vec[i], (0 * 0 + 1 * 1 + 2 * 2 + 3 * 3) * i);
      APPROX_EQ(d_vec[i],  (0 * 0 + 1 * 1 + 2 * 2 + 3 * 3) * i, 1e-7);
    }
    printf("done\n");
}