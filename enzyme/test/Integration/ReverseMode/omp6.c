//   note not doing O0 below as to ensure we get tbaa
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops %O0TBAA %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
//   note not doing O0 below as to ensure we get tbaa
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops %O0TBAA %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi

# include <stdio.h>
# include <stdlib.h>
#include <math.h>

#include "test_utils.h"

__attribute__((noinline))
void set(double *a, double x){
    a[0] = x;
}
void msg(double* in) {
    #pragma omp parallel for
    for (unsigned long long i=0; i<20; i++) {
        double m;
        set(&m, in[i]);
        in[i] = m * m;
    }
}

void __enzyme_autodiff(void*, ...);

int main ( int argc, char *argv[] ) {

  double array[20];
  for(int i=0; i<20; i++)
      array[i] = i+1;
  double darray[20];
  for(int i=0; i<20; i++)
      darray[i] = 1.0;
  
  __enzyme_autodiff((void*)msg, &array, &darray);
  
  for(int i=0; i<20; i++) {
      APPROX_EQ(darray[i], 2 * (i+1), 1e-10);
  }
  return 0;
}
