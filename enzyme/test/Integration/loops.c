// RUN: clang -std=c11 -O0 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -S | lli - 
// RUN: clang -std=c11 -O1 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -S | lli - 
// RUN: clang -std=c11 -O2 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -S | lli - 
// RUN: clang -std=c11 -O3 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -S | lli - 
// RUN: clang -std=c11 -O0 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_inline=1 -S | lli - 
// RUN: clang -std=c11 -O1 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_inline=1 -S | lli - 
// RUN: clang -std=c11 -O2 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_inline=1 -S | lli - 
// RUN: clang -std=c11 -O3 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_inline=1 -S | lli - 

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

#define __builtin_autodiff __enzyme_autodiff

double __enzyme_autodiff(void*, ...);

//float man_max(float* a, float* b) {
//  if (*a > *b) {
//    return *a;
//  } else {
//    return *b;
//  }
//}
void compute_loops(float* a, float* b, float* ret) {
  double sum0 = 0.0;
  for (int i = 0; i < 100; i++) {
    sum0 += *a + *b;
  }
  *ret = sum0;
}



int main(int argc, char** argv) {



  float a = 2.0;
  float b = 3.0;



  float da = 0;//(float*) malloc(sizeof(float));
  float db = 0;//(float*) malloc(sizeof(float));


  float ret = 0;
  float dret = 1.0;

  __builtin_autodiff(compute_loops, &a, &da, &b, &db, &ret, &dret);

  APPROX_EQ(da, 100*1.0f, 1e-10);
  APPROX_EQ(db, 100*1.0f, 1e-10);

  printf("hello! %f, res2 %f, da: %f, db: %f\n", ret, ret, da,db);
  return 0;
}
