// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "test_utils.h"

#define __builtin_autodiff __enzyme_autodiff
double __enzyme_autodiff(void*, ...);

__attribute__((noinline))
unsigned long long* substore(unsigned long long flt, unsigned long long integral) {
    unsigned long long* data = (unsigned long long*)malloc(2*sizeof(unsigned long long));

    double asflt;
    memcpy(&asflt, &flt, sizeof(asflt));

    *((double*)data) = asflt;
    data[1] = integral;
    return data;
}

double foo(double inp) {
  //union bitcaster bc;
  //bc.d = inp;
  unsigned long long res;
  memcpy(&res, &inp, sizeof(res));
  unsigned long long* data = substore(res, 3);
  return ((double*)data)[0];
}

double call(double inp) {
  return __builtin_autodiff(foo, inp);
}

int main() {
    for(int i=-20; i<=20; i++) {
        printf("i=%d\n", i);
        APPROX_EQ(call(i/10.), 1.0, 1e-10);
    }
}
