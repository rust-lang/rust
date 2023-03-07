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

#include <stdlib.h> 
#include <stdio.h> 
#include <stdbool.h>
#include <math.h> 


extern double __enzyme_autodiff(void*, double);

// testing both std::wcout and std::locale
// https://en.cppreference.com/w/cpp/locale/locale
double fn(double vec) {
  std::wcout.put('f');
  std::wcout << 1 << 1.0 << "somerandomchars";
  std::locale::global(std::locale(""));
  std::wcout.sync_with_stdio();
  std::wcout.imbue(std::locale());

  // Currently not working
  // std::wcout << "User-preferred locale setting is ";
  // std::wcout << std::locale("").name().c_str();
  // std::wcout << 1000.01 << '\n';
  // std::wcout << 1000.01 << std::endl;
  // std::wcout << 1000.01 << std::flush;

  return vec * vec;
}

int main() {
    double x = 2.1;
    double dsq = __enzyme_autodiff((void*)fn, x);

    APPROX_EQ(dsq, 2 * x, 1e-7);
}

