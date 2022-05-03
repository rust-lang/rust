// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -

#include "test_utils.h"

extern double __enzyme_fwddiff(double (*)(double, double), double,...);

double fun(double x1, double x2) {
    double w1 = x1;
    double w2 = x2;
    double w3 = w1 * w2;
    double w4 = 1.0 / w1;
    double w5 = w3 + w4;
    return w5;
}

double dfun(double x1, double x2) {
    return __enzyme_fwddiff(fun, x1, 1.0, x2, 0.0);
}

int main() {
    double res[] = {0.0,3.75,8.8888888888888893,15.9375};

    for(int i=1; i<5; i++) {
        double x = (double) i;
        double dfx = dfun(x,x*x);
        APPROX_EQ(dfx, res[i-1], 1e-10);
    } 
}
