// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/*
#ifdef __cplusplus
extern "C" {
#endif
static inline bool approx_fp_equality_float(float f1, float f2, double
threshold) { if (fabs(f1-f2) > threshold) return false; return true;
}

static inline bool approx_fp_equality_double(double f1, double f2, double
threshold) { if (fabs(f1-f2) > threshold) return false; return true;
}
#ifdef __cplusplus
}
#endif
*/

#define APPROX_EQ(LHS, RHS, THRES)                                             \
  {                                                                            \
    if (__builtin_fabs(LHS - RHS) > THRES) {                                   \
      fprintf(stderr,                                                          \
              "Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d " \
              "(%s)\n",                                                        \
              #LHS, LHS, #RHS, RHS, THRES, __FILE__, __LINE__,                 \
              __PRETTY_FUNCTION__);                                            \
      abort();                                                                 \
    }                                                                          \
  };


typedef struct {
  double dx, dy;
} Gradients;

extern Gradients __enzyme_fwddiff(double (*)(double, double), ...);
extern int enzyme_width;

double fun(double x1, double x2) {
  double w1 = x1;
  double w2 = x2;
  double w3 = w1 * w2;
  double w4 = 1.0 / w1;
  double w5 = w3 + w4;
  return w5;
}

Gradients dfun(double x, double y) {
  return __enzyme_fwddiff(fun, enzyme_width, 2, x, 1.0, 0.0, y, 0.0, 1.0);
}

int main() {
  double xres[] = {0.0, 3.75, 8.8888888888888893, 15.9375};
  double yres[] = {1.0, 2.0, 3.0, 4.0};

  for (int i = 1; i < 5; i++) {
    double x = (double)i;
    Gradients df = dfun(x, x * x);
    APPROX_EQ(df.dx, xres[i - 1], 1e-10);
    APPROX_EQ(df.dy, yres[i - 1], 1e-10);
  }
}
