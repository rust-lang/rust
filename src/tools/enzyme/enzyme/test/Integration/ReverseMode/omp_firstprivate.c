//   note not doing O0 below as to ensure we get tbaa
// TODO: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 %O0TBAA -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O1 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O2 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O3 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
//   note not doing O0 below as to ensure we get tbaa
// TODO: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 %O0TBAA -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O1 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O2 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -O3 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi


extern int omp_get_max_threads();
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

double __enzyme_autodiff(void*, ...);

void omp(double* a, double in, int N) {
  #pragma omp parallel for firstprivate(in)
  for (int i=0; i<N; i++) {
    a[i] = in;
    in = 0;
  }
}

int main(int argc, char** argv) {

  int N = 20;
  double a[N];
  double d_a[N];

  for(int i=0; i<N; i++)
    d_a[i] = 1.0f;
  
  //omp(*a, N);
  printf("ran omp\n");
  double res = __enzyme_autodiff((void*)omp, a, d_a, (double)1.0f, N);

  for(int i=0; i<N; i++) {
    printf("a[%d]=%f  d_a[%d]=%f\n", i, a[i], i, d_a[i]);
  }

  double expected = omp_get_max_threads();
  if (expected > N) expected = N;
  
  APPROX_EQ(res, expected, 1e-10);
  return 0;
}
