// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S

#include "test_utils.h"
#include <iostream>
#include <sstream>
#include <utility>

extern double __enzyme_autodiff(void*, double);

double fn(double vec) {
  std::stringstream testInput("14 1.5 somerandomextrachars");
  double in;
  float in2;
  testInput >> in >> in2;

  testInput.ignore();

  char ch;
  testInput.get(ch);

  char foo[5];
  const char fdelim = '\t';
  testInput.get(foo, 3, fdelim);

  // The following two lines cause a segfault with Enzyme
  // char bar[5];
  // testInput.getline(bar, 3);

  return vec * vec * in * in2;
}

int main() {
    double x = 2.1;
    double dsq = __enzyme_autodiff((void*)fn, x);

    APPROX_EQ(dsq, 14 * 1.5 * 2 * x, 1e-7);
}
