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
