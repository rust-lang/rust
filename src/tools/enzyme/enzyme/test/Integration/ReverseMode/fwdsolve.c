// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"


void __enzyme_autodiff(void*, ...);

__attribute__((noinline))
void forward_sub(int N, double* __restrict__ L, double * __restrict__ b, double * __restrict__ out) {
    /*
      """x = forward_sub(L, b) is the solution to L x = b
       L must be a lower-triangular matrix
       b must be a vector of the same leading dimension as L
    """
    */
    for (size_t i=0; i<N; i++) {
        double tmp = b[i];
        if (i > 1)
        for (size_t j=0; j<i-1; j++) {
            tmp -= L[i*N+j] * out[j];
        }
        out[i] = tmp;
        // assumed to be 1
        // out[i] = tmp / L[i*N+i];
    }
}

int main() {

  double L[9];
  double dL[9];
  for (int i=0; i<9; i++) {
      L[i] = i+1;
      dL[i] = 0;
  }
  double b[3];
  double db[3];
  double out[3];
  double dout[3];
  for (int i=0; i<3; i++) {
      b[i] = (i+1) / 10.;
      db[i] = 0;
      dout[i] = 10000 * (i + 1);
  }

  forward_sub(3, L, b, out);

  for (int i=0; i<9; i++) {
      L[i] = i;
  }
  __enzyme_autodiff(forward_sub, 3, L, dL, b, db, out, dout);

  for (int i=0; i<9; i++) {
    printf("dL[%d]=%f\n", i, dL[i]);
  }

  for (int i=0; i<3; i++) {
    printf("dB[%d]=%f\n", i, db[i]);
  }

  double real_dL[9] = { 0. };
  real_dL[6] = -3000;
  for (int i=0; i<9; i++) {
    APPROX_EQ(dL[i], real_dL[i], 1e-10);
  }

  double real_db[3] = { -170000., 20000, 30000 };
  for (int i=0; i<3; i++) {
    APPROX_EQ(db[i], real_db[i], 1e-10);
  }

  return 0;
}
