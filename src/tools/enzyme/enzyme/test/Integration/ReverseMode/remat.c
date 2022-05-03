// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli - 

// test.c
#include <stdio.h>
#include <stdlib.h>

#include "test_utils.h"

extern void __enzyme_autodiff(void*, ...);
void square(double** p_delv, double** p_e, int ** idx, int numReg, int numElemReg) {
    double* delv = *p_delv;
    double* e = *p_e;
    for (int r = 0; r < numReg; r++) {
        double* tmp = (double*)malloc(numElemReg * sizeof(double));
        for (int i=0; i<numElemReg; i++) {
            int off = idx[r][i];
            tmp[i] = delv[off];
        }
        for (int i=0; i<numElemReg; i++) {
            int off = idx[r][i];
            e[off] = tmp[i] * tmp[i];
        }
        free(tmp);
    }
}
int main() {
    int numReg = 100;
    double *delv = (double*)malloc(sizeof(double)*numReg);
    double *e = (double*)malloc(sizeof(double)*numReg);
    double *d_delv = (double*)malloc(sizeof(double)*numReg);
    double *d_e = (double*)malloc(sizeof(double)*numReg);
    int* idxs[numReg];
    int numRegElem = 200;
    for (int i=0; i<numReg; i++) {
        int* data = (int*)malloc(sizeof(int)*numRegElem);
        for (int j=0; j<numRegElem; j++) {
            data[j] = j % numReg;
        }
        idxs[i] = data;
        delv[i] = i;
        d_delv[i] = 0;
        e[i] = 0;
        d_e[i] = 1;
    }
    
    square(&delv, &e, idxs, numReg, numRegElem);
    for (int i=0; i<numReg; i++) {
        printf("e=%f delv=%f\n", e[i], delv[i]);
    }

    __enzyme_autodiff((void*)square, &delv, &d_delv, &e, &d_e, idxs, numReg, numRegElem);
    for (int i=0; i<numReg; i++) {
        printf("d_e=%f d_delv=%f\n", d_e[i], d_delv[i]);
        APPROX_EQ(d_e[i], 0.0, 1e-10);
        APPROX_EQ(d_delv[i], 2.0 * i, 1e-10);
    }
}

