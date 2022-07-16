// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -O1 -g %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -

#include <stdio.h>

#include "test_utils.h"

extern void __enzyme_batch(void *, int, int, char *, char *, char *, char *);
extern int enzyme_dup;
extern int enzyme_width;

#pragma pack(1)
struct Foo {
  int arr[3];
  double x;
  float y;
  double res;
};

void f(char *foo) {
  double *xptr = (double *)(foo + sizeof(int[3]));
  float *yptr = (float *)(foo + sizeof(int[3]) + sizeof(double));
  double *resptr =
      (double *)(foo + sizeof(int[3]) + sizeof(double) + sizeof(float));
  double x = *xptr;
  float y = *yptr;
  *resptr = x * y;
}

void df(char *dfoo1, char *dfoo2, char *dfoo3, char *dfoo4) {
  __enzyme_batch((void *)f, enzyme_width, 4, dfoo1, dfoo2, dfoo3, dfoo4);
}

int main() {
  Foo foo1;
  foo1.x = 10;
  foo1.y = 9.0;
  Foo foo2;
  foo2.x = 99.0;
  foo2.y = 7.0;
  Foo foo3;
  foo3.x = 1.1;
  foo3.y = 9.0;
  Foo foo4;
  foo4.x = 3.14;
  foo4.y = 0.1;

  double expected[4] = {90.0, 693.0, 9.9, 0.314};

  df((char *)&foo1, (char *)&foo2, (char *)&foo3, (char *)&foo4);

  APPROX_EQ(foo1.res, expected[0], 1e-9);
  APPROX_EQ(foo2.res, expected[1], 1e-9);
  APPROX_EQ(foo3.res, expected[2], 1e-9);
  APPROX_EQ(foo4.res, expected[3], 1e-8);
}