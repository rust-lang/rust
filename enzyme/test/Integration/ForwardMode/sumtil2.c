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
    ret += vec[i];
    if (ret > 15) break;
    ret += vec[i];
  }
  return ret;
}

int main() {
    double vec[] = {1, 2., 3., 4., 5.};
    double d_vec[] = {2., 0., 1., 9., 8.};
    double d_start = __enzyme_fwddiff(sumtil, vec, d_vec, 5);
          
    printf("d_start=%f\n", d_start);
    APPROX_EQ(d_start, 15.0 , 1e-10);
    
    printf("done\n");
}