// RUN: %clang++ -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

//#include <math.h>

#include "test_utils.h"

double __enzyme_autodiff1(void*, int, double, double);
double __enzyme_autodiff2(void*, double);

static double f(double x, double y) {
    return x * y;
}

extern int enzyme_const;

static double df(double x) {
    return x * __enzyme_autodiff1((void*)f, enzyme_const, x, 3.0);
}

int main(int argc, char** argv) {
  double ret = __enzyme_autodiff2((void*)df, 5.0);
  printf("ret=%f\n");
  APPROX_EQ(ret, 10.0, 1e-7);

  return 0;
}
