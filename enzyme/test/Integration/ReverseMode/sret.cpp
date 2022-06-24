// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S

#include "test_utils.h"
#include <iostream>
#include <sstream>
#include <utility>

typedef struct {
    double df[3];
} Gradient;
extern Gradient __enzyme_autodiff(void*, double, double , double);

double myfunction(double x, double y, double z){
    return x * y * z;
}

void dmyfunction(double x, double y, double z, double* res) {
    Gradient g = __enzyme_autodiff((void*)myfunction, x, y, z);

    res[0]=g.df[0];
    res[1]=g.df[1];
    res[2]=g.df[2];
}

int main() {
    double *res=new double(3);
    dmyfunction(3,4,5,res);
    APPROX_EQ(res[0], 4*5., 1e-7);
    APPROX_EQ(res[1], 3*5., 1e-7);
    APPROX_EQ(res[1], 3*4., 1e-7);
}
