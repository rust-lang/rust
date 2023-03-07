// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 

#include <stdint.h>
#include <math.h>
#include <stdio.h>

#include "test_utils.h"

double f(double x) {
  int exp;
  return frexp(x, &exp);
}

double __enzyme_autodiff(void*, ...);

int main() {

    double data[] = { 0.9, 1.0, 1.2, 2.0, 2.3, 32.0, 33.0, 0.25, 0.26, -0.9, -1.0, -1.2, -2.0, -2.3, 32.0, -33.0};

    for (int i=0; i<sizeof(data)/sizeof(*data); i++) {
        double truev = f(data[i])/data[i];
        double ad =  __enzyme_autodiff((void*)f, data[i]);
        printf("x=%f f(x)=%f f/x=%f, d/dx=%f\n", data[i], f(data[i]), truev, ad);
        APPROX_EQ(ad, truev, 1e-10);
    }
}

