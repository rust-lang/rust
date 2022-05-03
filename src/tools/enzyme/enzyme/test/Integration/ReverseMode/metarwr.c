// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

void __enzyme_autodiff(void*, ...);

void call(double* __restrict__  a, long** data) {
  long* segment = data[0];
  long size = segment[1] - segment[0];
  printf("seg[1]=%d seg[0]=%d\n", segment[1], segment[0]);
  for (size_t i=0; i<size; i++)
    a[i] *= 2;
  data[0] = 0;
}

void alldiv(double* __restrict__ a, long** meta) {
  call(a, meta);
  a[0] = 0;
}

int main(int argc, char** argv) {

  long meta[2] = { 198, 200 }; 
  long* mmeta = (long*)meta;
  double *val = malloc(200*sizeof(double));
  val[1] = 7;
  double *dval = malloc(200*sizeof(double));
  dval[1] = 1;
  double* a = (double*)val;
  double* da = (double*)dval;
  
  __enzyme_autodiff((void*)alldiv, (double*)val, (double*)dval, &mmeta);

  printf("a = %p, da=%p\n", a, da);
  printf("val=%f dval=%f\n", val[0], dval[0]);
  printf("meta=%d\n", meta);
  fflush(0);

  APPROX_EQ(dval[1], 2.0, 1e-8);
  return 0;
}
