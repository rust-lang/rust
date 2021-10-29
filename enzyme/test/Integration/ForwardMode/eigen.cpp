// RUN: %clang++ -mllvm -force-vector-width=1 -ffast-math -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

#include "test_utils.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

double __enzyme_fwddiff(double(double), double, double);

double square(double x) {
  Eigen::Vector3d v(x, x * x, x * x * x);
  v *= 2;
  return v[1];
}

double dsquare(double x) { return __enzyme_fwddiff(square, x, 1.0); }

int main() {
  double x = 4;
  double res = dsquare(x);
  APPROX_EQ(res, 16.0, 1e-10);
  printf("dsquare(%f)=%f\n", x, res);
  return 0;
}