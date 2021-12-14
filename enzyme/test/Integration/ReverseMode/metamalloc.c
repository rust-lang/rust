// RUN: %clang -std=c11 -Xclang -new-struct-path-tbaa -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -Xclang -new-struct-path-tbaa -fno-unroll-loops -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -Xclang -new-struct-path-tbaa -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -Xclang -new-struct-path-tbaa -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -Xclang -new-struct-path-tbaa -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -Xclang -new-struct-path-tbaa -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -Xclang -new-struct-path-tbaa -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -Xclang -new-struct-path-tbaa -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

double __enzyme_autodiff(void*, ...);

struct {
    int count;
void* (*allocfn)(long int);
} tup = {0, malloc};
__attribute__((noinline))
void* metamalloc(long int size) {
    void* ret = tup.allocfn(size);
    //if (ret != 0)
    //  tup.count++;
    return ret;
}
__attribute__((noinline))
void square(double* x) {
    *x *= *x;
}
double alldiv(double x) {
  double* mem = (double*)metamalloc(8);
  *mem = x;
  square(mem);
  return mem[0];
}


static void* (*sallocfn)(int) = malloc;
__attribute__((noinline))
void* smetamalloc(int size) {
    return sallocfn(size);
}
double salldiv(double x) {
  double* mem = (double*)metamalloc(8);
  *mem = x * x;
  return mem[0];
}

int main(int argc, char** argv) {
  double res = __enzyme_autodiff((void*)alldiv, 3.14);
  APPROX_EQ(res, 6.28, 1e-6);
  double sres = __enzyme_autodiff((void*)salldiv, 3.14);
  APPROX_EQ(sres, 6.28, 1e-6);
  return 0;
}
