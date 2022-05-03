// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S

#include <iostream>
#include "test_utils.h"

extern double __enzyme_autodiff(void*, double);

double fn(double vec) {
  std::cout << "hello" << 7 << '7' << std::endl;
  std::cout << "foo" << std::flush;
  return vec * vec;
}

int main() {
    double x = 2.1;
    double dsq = __enzyme_autodiff((void*)fn, x);

    APPROX_EQ(dsq, 2 * x, 1e-7);
}
