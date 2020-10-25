// RUN: %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - -g | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O1 %s -S -emit-llvm -o - -g | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O2 %s -S -emit-llvm -o - -g | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O3 %s -S -emit-llvm -o - -g | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - -g | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O1 %s -S -emit-llvm -o - -g | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O2 %s -S -emit-llvm -o - -g | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O3 %s -S -emit-llvm -o - -g | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

//#include <math.h>

#include "test_utils.h"

#define __builtin_autodiff __enzyme_autodiff

double __enzyme_autodiff(void*, double, unsigned);

static double taylorlog(double x, unsigned SINCOSN) {
  double sum = 0;
  for(int i=1; i<=SINCOSN; i++) {
    sum += __builtin_pow(x, i);
  }
  return sum;
}

int main(int argc, char** argv) {
  
  double ret = __builtin_autodiff(taylorlog, 0.5, 10000);

  APPROX_EQ(ret, 4.0, 1e-7);

  return 0;
}
