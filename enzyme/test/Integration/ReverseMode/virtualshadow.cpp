// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -

#include <stdio.h>
#include "test_utils.h"
struct S {
   double (*fn)(double);
   double val;
};

double square(double x){ return x * x; }

double foo(struct S* s) {
  return square(s->val);
}

void primal() {
  struct S s;
  s.fn = square;
  s.val = 3.0;
  printf("%f\n", foo(&s));
}

struct DoubleAndTape {
    double x;
    void* tape;
};

DoubleAndTape __enzyme_augmentfwd(void*, double);
DoubleAndTape fwdsquare(double x) {
    return __enzyme_augmentfwd((void*)square, x);
}
double __enzyme_reverse(void*, double, double, void*);
double revsquare(double x, double differet, void* tape) {
    return __enzyme_reverse((void*)square, x, differet, tape);
}
void* shadow_sq[2] = {(void*)fwdsquare, (void*)revsquare};
void __enzyme_autodiff(void*, void*, void*);
void reverse() {
  struct S s;
  s.fn = square;
  s.val = 3.0;
  struct S d_s;
  d_s.fn = (double (*)(double))shadow_sq;
  d_s.val = 0.0;
  __enzyme_autodiff((void*)foo, &s, &d_s);
  printf("shadow res=%f\n", d_s.val);
  APPROX_EQ(d_s.val, 6.0, 1e-7);
}

int main() {
    primal();
    reverse();
}
