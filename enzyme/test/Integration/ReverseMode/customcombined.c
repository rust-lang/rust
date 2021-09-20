// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

#include "test_utils.h"

double __enzyme_autodiff(void*, double);

__attribute__((noinline))
void square_(const double* src, double* dest) {
    *dest = *src * *src;
}

int augment = 0;
void* augment_square_(const double* src, const double *d_src, double* dest, double* d_dest) {
    augment++;
    // intentionally incorrect for debugging
    *dest = 7.0;
    *d_dest = 11.0;
    return NULL;
}

int gradient = 0;
void gradient_square_(const double* src, double *d_src, const double* dest, const double* d_dest, void* tape) {
    gradient++;
    // intentionally incorrect for debugging
    *d_src = 13.0;
}

void* __enzyme_register_gradient_square[] = {
    (void*)square_,
    (void*)augment_square_,
    (void*)gradient_square_,
};


double square(double x) {
    double y;
    square_(&x, &y);
    return y;
}

double dsquare(double x) {
    return __enzyme_autodiff((void*)square, x);
}


int main() {
    double res = dsquare(3.0);
    printf("res=%f augment=%d gradient=%d\n", res, augment, gradient);
    APPROX_EQ(res, 13.0, 1e-10);
    APPROX_EQ(augment, 1.0, 1e-10);
    APPROX_EQ(gradient, 1.0, 1e-10);
}
