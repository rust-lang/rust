// RUN: %clang++ -ffast-math -O1 -Xclang -disable-llvm-optzns -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -fno-exceptions %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -ffast-math -O1 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -fno-exceptions %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -ffast-math -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -fno-exceptions %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -ffast-math -O3 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -fno-exceptions %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -ffast-math -O1 -Xclang -disable-llvm-optzns -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -fno-exceptions %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -ffast-math -O1 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -fno-exceptions %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -ffast-math -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -fno-exceptions %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -ffast-math -O3 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -fno-exceptions %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <math.h>

#include <vector>

#include "test_utils.h"

extern void __enzyme_autodiff(void*, double*, double*, int);
/*double max(double x, double y) {
    return (x > y) ? x : y;
}*/

double reduce_max(double* vec, int size) {
  double ret = -INFINITY;
  std::vector<double> maxes;
  for (int i = 0; i < size; i++) {
    if (vec[i] > ret) {
      maxes.clear();
      ret = vec[i];
    }
    if (vec[i] == ret) {
      maxes.push_back(vec[i]);
    }
  }
  ret = 0;
  for(auto a : maxes) {
    ret += a;
  }
  ret /= maxes.size();
  return ret;
}

int main() {
    double vec[] = {-1., 2., -0.2, 2., 1.};
    double d_vec[] = {0., 0., 0., 0., 0.};
    double max_val = reduce_max(vec, 5);
    printf("reduce_max=%f\n", max_val);
    __enzyme_autodiff((void*)reduce_max, vec, d_vec, 5);
    for(int i=0; i<5; i++) {
       printf("d_reduce_max(%i)=%f\n", i, d_vec[i]);
    }

    double ans[] = {0, 0.5, 0, 0.5, 0};
    for(int i=0; i<5; i++) {
      APPROX_EQ(d_vec[i], ans[i], 1e-10);
    }
}
