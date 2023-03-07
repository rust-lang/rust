// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -

#include <stdio.h>
#include <math.h>
#include <vector>
#include <assert.h>

#include "test_utils.h"

extern void __enzyme_autodiff(void*, std::vector<double>*, std::vector<double>*);
/*double max(double x, double y) {
    return (x > y) ? x : y;
}*/

double reduce_max(const std::vector<double>& vec) {
  double ret = -INFINITY;
  for (size_t i = 0, size = vec.size(); i < size; i++) {
    ret = (ret > vec[i]) ? ret : vec[i];
  }
  return ret;
}

int main() {
    std::vector<double> vec = {-1., 2., -0.2, 2.2, 1.};
    std::vector<double> d_vec = {0., 0., 0., 0., 0.};
    double max_val = reduce_max(vec);
    printf("reduce_max=%f\n", max_val);
    __enzyme_autodiff((void*)reduce_max, &vec, &d_vec);

    for(int i=0; i<5; i++) {
      printf("d_reduce_max(%i)=%f\n", i, d_vec[i]);
    }

    double ans[] = {0, 0, 0, 1, 0};
    for(int i=0; i<5; i++) {
      printf("i=%d d_vec=%f ans=%f\n", i, d_vec[i], ans[i]);
      APPROX_EQ(d_vec[i], ans[i], 1e-7);
    }
    printf("done\n");
}