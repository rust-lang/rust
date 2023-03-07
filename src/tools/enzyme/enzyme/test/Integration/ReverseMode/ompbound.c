//   note not doing O0 below as to ensure we get tbaa
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 %O0TBAA -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O1 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O2 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O3 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
//   note not doing O0 below as to ensure we get tbaa
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 %O0TBAA -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O1 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O2 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O3 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

double __enzyme_autodiff(void*, ...);

/*
void omp(float& a, int N) {
  #define N 20
  #pragma omp parallel for
  for (int i=0; i<N; i++) {
    //a[i] *= a[i];
    (&a)[i] *= (&a)[i];
  }
  #undef N
  (&a)[0] = 0;
}
*/
void omp(float* a, int N) {
  #pragma omp parallel for
  for (int i=0; i<N; i++) {
    //a[i] *= a[i];
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
  
  //omp(*a, N);
  printf("ran omp\n");
  __enzyme_autodiff((void*)omp, a, d_a, N);

  for(int i=0; i<N; i++) {
    printf("a[%d]=%f  d_a[%d]=%f\n", i, a[i], i, d_a[i]);
  }

  //APPROX_EQ(da, 17711.0*2, 1e-10);
  //APPROX_EQ(db, 17711.0*2, 1e-10);
  //printf("hello! %f, res2 %f, da: %f, db: %f\n", ret, ret, da,db);
  APPROX_EQ(d_a[0], 0.0f, 1e-10);
  for(int i=1; i<N; i++) {
    APPROX_EQ(d_a[i], 2.0f*(i+1), 1e-10);
  }
  return 0;
}
