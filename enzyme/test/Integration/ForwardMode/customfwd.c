
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

#include "test_utils.h"

double __enzyme_fwddiff(void*, ...);

__attribute__((noinline))
void square_(const double* src, double* dest) {
    *dest = *src * *src;
}

int derivative = 0;
void derivative_square_(const double* src, const double *d_src, const double* dest, double* d_dest) {
    derivative++;
    // intentionally incorrect for debugging
    *d_dest = 100; 
}

void* __enzyme_register_derivative_square[] = {
    (void*)square_,
    (void*)derivative_square_,
};


double square(double x) {
    double y;
    square_(&x, &y);
    return y;
}

double dsquare(double x) {
    return __enzyme_fwddiff((void*)square, x, 1.0);
}


int main() {
    double res = dsquare(3.0);
    printf("res=%f derivative=%d\n", res, derivative);
    APPROX_EQ(res, 100.0, 1e-10);
    APPROX_EQ(derivative, 1.0, 1e-10);
}