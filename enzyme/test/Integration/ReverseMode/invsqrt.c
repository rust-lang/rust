// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "test_utils.h"

// Fast inverse sqrt
// Code taken from https://en.wikipedia.org/wiki/Fast_inverse_square_root
float Q_rsqrt( float number )
{
  long i;
  float x2, y;
  const float threehalfs = 1.5F;

  x2 = number * 0.5F;
  y  = number;
  i  = * ( long * ) &y;                       // evil floating point bit level hacking
  i  = 0x5f3759df - ( i >> 1 );               // what the [...]?
  y  = * ( float * ) &i;
  y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
  return y;
}


double invmag(double* __restrict__ A, int n) {
  double sumsq = 0;
  for (int i=0; i<n; i++) {
    sumsq += A[i] * A[i];
  }
  return Q_rsqrt(sumsq);
}



// Returns { optional tape, original return (if pointer), shadow return (if pointer) }
void aug_rsqrt(float x) {
  // Nothing need to be done in augmented forward pass
}

// Arguments: all pointers duplicated, gradient of the return, tape (if provided)
float rev_rsqrt(float x, float grad_out) {
  // derivative of x^(-1/2) = -1/2 x^(-3/2)
  return -grad_out * Q_rsqrt(x) / (2 * x);
}

void* __enzyme_register_gradient_rsqrt[3] = { (void*)Q_rsqrt, (void*)aug_rsqrt, (void*)rev_rsqrt };

void __enzyme_autodiff(void*, ...);

int main(int argc, char *argv[]) {
  int n = 3;

  double *A = (double*)malloc(sizeof(double) * n);
  for(int i=0; i<n; i++)
   A[i] = (i+1);
  assert(A != 0);

  double *grad_A = (double*)malloc(sizeof(double) * n);
  assert(grad_A != 0);
  for(int i=0; i<n; i++)
   grad_A[i] = 0;
 
  __enzyme_autodiff((void*)invmag, A, grad_A, n);

  double im = invmag(A, n);
  im = im*im*im;

  for(int i=0; i<n; i++)
    printf("A[%d]=%f dA[%d]=%f\n", i, A[i], i, grad_A[i]);

  fflush(0);
  
  for(int i=0; i<n; i++)
    APPROX_EQ(grad_A[i], -A[i]*im, 1e-3); 

  return 0;
}
