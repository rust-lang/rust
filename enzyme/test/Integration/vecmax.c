// RUN: if [ %llvmver -le 11 ]; then %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - ; fi
// RUN: if [ %llvmver -le 9 ]; then %clang -std=c11 -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - ; fi
// RUN: if [ %llvmver -le 8 ]; then %clang -std=c11 -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - ; fi
// RUN: if [ %llvmver -le 8 ]; then %clang -std=c11 -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - ; fi
// RUN: if [ %llvmver -le 11 ]; then %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -le 9 ]; then %clang -std=c11 -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -le 8 ]; then %clang -std=c11 -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -le 8 ]; then %clang -std=c11 -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -; fi

#include <stdio.h>
#include <math.h>

#include "test_utils.h"

extern void __enzyme_autodiff(void*, double*, double*, int);
/*double max(double x, double y) {
    return (x > y) ? x : y;
}*/

double reduce_max(double* vec, int size) {
  double ret = -INFINITY;
  for (int i = 0; i < size; i++) {
    ret = (ret > vec[i]) ? ret : vec[i];
  }
  return ret;
}

int main() {
    double vec[] = {-1., 2., -0.2, 2., 1.};
    double d_vec[] = {0., 0., 0., 0., 0.};
    double max_val = reduce_max(vec, 5);
    printf("reduce_max=%f\n", max_val);
    __enzyme_autodiff(reduce_max, vec, d_vec, 5);

    for(int i=0; i<5; i++) {
      printf("d_reduce_max(%i)=%f\n", i, d_vec[i]);
    }

    double ans[] = {0, 0, 0, 1, 0};
    for(int i=0; i<5; i++) {
      printf("i=%d d_vec=%f ans=%f\n", i, d_vec[i], ans[i]);
      APPROX_EQ(d_vec[i], ans[i], 1e-7);
    }
    printf("done\n");
}