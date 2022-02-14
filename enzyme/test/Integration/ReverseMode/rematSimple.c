// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - ; fi

// test.c
#include <stdio.h>
#include <stdlib.h>

#include "test_utils.h"

extern void __enzyme_autodiff(void*, ...);
void square(double* __restrict__ delv, double* __restrict__ e, unsigned long long numReg) {
    for (unsigned long long r = 0; r < 20; r++) {
        double* tmp = (double*)malloc(numReg * sizeof(double));
        for (unsigned long long i=0; i<numReg; i++) {
            tmp[i] = delv[i];
        }
        for (unsigned long long i=0; i<numReg; i++) {
            e[i] += tmp[i] * tmp[i];
        }
        free(tmp);
    }
}
int main() {
    unsigned long long numReg = 100;
    double *delv = (double*)malloc(sizeof(double)*numReg);
    double *e = (double*)malloc(sizeof(double)*numReg);
    double *d_delv = (double*)malloc(sizeof(double)*numReg);
    double *d_e = (double*)malloc(sizeof(double)*numReg);
    
    for (int i=0; i<numReg; i++) {
        delv[i] = i;
        d_delv[i] = 0;
        e[i] = 0;
        d_e[i] = 1;
    }
    
    square(delv, e, numReg);
    for (int i=0; i<numReg; i++) {
        printf("e=%f delv=%f\n", e[i], delv[i]);
    }

    __enzyme_autodiff((void*)square, delv, d_delv, e, d_e, numReg);
    for (int i=0; i<numReg; i++) {
        printf("d_e[%d]=%f d_delv=%f\n", i, d_e[i], d_delv[i]);
    }
    for (int i=0; i<numReg; i++) {
        APPROX_EQ(d_e[i], 1.0, 1e-10);
        APPROX_EQ(d_delv[i], 2.0 * i * 20, 1e-10);
    }
    free(delv);
    free(e);
    free(d_delv);
    free(d_e);
}

