//   note not doing O0 below as to ensure we get tbaa
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops %O0TBAA %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
//   note not doing O0 below as to ensure we get tbaa
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops %O0TBAA %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi

# include <stdlib.h>
# include <stdio.h>


void msg(double* in, int *len, unsigned int slen) {
    if (slen != 0) {
    #pragma omp parallel for firstprivate(slen)
    for (unsigned int i=0; i<slen; i++) {
/*
        int L = len[i] / 2;
        __builtin_assume(L > 0);
        for(int j=0; j<L; j++)
            in[j*10+i] *= L;
        len[i] = 0;
        */
    }
    }
}
void __enzyme_autodiff(void*, ...);

int main ( int argc, char *argv[] ) {

  double array[200];
  double darray[200];
  int len[10] = {20};
  int slen = 10;
  __enzyme_autodiff((void*)msg, &array, &darray, &len, slen);
     return 0;
}
