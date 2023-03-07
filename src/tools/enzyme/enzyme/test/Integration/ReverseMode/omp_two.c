//   note not doing O0 below as to ensure we get tbaa
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 %O0TBAA -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O1 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O2 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O3 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
//   note not doing O0 below as to ensure we get tbaa
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 %O0TBAA -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O1 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O2 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O3 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

void __enzyme_autodiff(void*, ...);

void omp(float* a, float *b, int N, int M) {

  #pragma omp parallel
  {
    
    #pragma omp for
    for (int i=0; i<N; i++) {
      a[i] *= a[i];
    }
    
    #pragma omp for
    for (int i=0; i<M; i++) {
      b[i] *= b[i];
    }

  }

  return;
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

  int M = 40;
  float b[M];
  for(int i=0; i<M; i++) {
    b[i] = 0.1*i+1;
  }

  float d_b[M];
  for(int i=0; i<M; i++)
    d_b[i] = 1.0f;
  
  //omp(a, b, N, M);
  printf("ran omp\n");
  __enzyme_autodiff((void*)omp, a, d_a, b, d_b, N, M);

  for(int i=0; i<N; i++) {
    printf("a[%d]=%f  d_a[%d]=%f\n", i, a[i], i, d_a[i]);
  }
  
  for(int i=0; i<M; i++) {
    printf("b[%d]=%f  d_b[%d]=%f\n", i, b[i], i, d_b[i]);
  }

  for(int i=0; i<N; i++) {
    APPROX_EQ(d_a[i], 2.0f*(i+1), 1e-10);
  }
  for(int i=0; i<M; i++) {
    APPROX_EQ(d_b[i], 2.0f*(0.1*i+1), 1e-5);
  }
  return 0;
}
