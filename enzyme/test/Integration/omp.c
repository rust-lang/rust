//   note not doing O0 below as to ensure we get tbaa
// RUN: %clang -fopenmp -std=c11 -O1 -disable-llvm-optzns %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -fopenmp -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -fopenmp -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -fopenmp -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
//   note not doing O0 below as to ensure we get tbaa
// RUN: %clang -fopenmp -std=c11 -O1 -Xclang -disable-llvm-optzns %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -fopenmp -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -fopenmp -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -fopenmp -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

double __enzyme_autodiff(void*, ...);

void omp(float* a, int N) {
  #pragma omp parallel for
  for (int i=0; i<N; i++) {
    a[i] *= a[i];
  }
  a[0] = 0;
}

int main(int argc, char** argv) {

  int N = 20;
  float a[N];
  for(int i=0; i<N; i++) {
    a[i] = i+1;
  }

  float d_a[N];
  for(int i=0; i<N; i++)
    d_a[i] = 1.0f;
    
  __enzyme_autodiff(omp, a, d_a, N);

  //APPROX_EQ(da, 17711.0*2, 1e-10);
  //APPROX_EQ(db, 17711.0*2, 1e-10);
  //printf("hello! %f, res2 %f, da: %f, db: %f\n", ret, ret, da,db);
  return 0;
}
