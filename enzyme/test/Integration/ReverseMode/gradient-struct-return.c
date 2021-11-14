// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include "test_utils.h"

typedef struct {
    double dx,dy;
} Gradients;

extern Gradients __enzyme_autodiff(void*, ...);

double mul(double x, double y) {
    return x * y;
}
Gradients dmul(double x, double y) {
    return __enzyme_autodiff((void*)mul, x, y);
}
int main() {
    double x = 1.0;
    double y = 2.0;
    printf("mul(x=%f,y%f)=%f\n", x, y, mul(x,y));
    printf("ddx dmul(x=%f,y%f)=%f\n", x, y, dmul(x,y).dx);
    printf("ddy dmul(x=%f,y%f)=%f\n", x, y, dmul(x,y).dy);
    APPROX_EQ(dmul(x,y).dx, 2.0, 10e-10);
    APPROX_EQ(dmul(x,y).dy, 1.0, 10e-10);
}
