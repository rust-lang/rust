#include <stdlib.h> 
#include <stdbool.h>
#include <math.h> 
 bool approx_fp_equality_float(float f1, float f2, double threshold) {
  if (fabs(f1-f2) > threshold) return false;
  return true;
}

static bool approx_fp_equality_double(double f1, double f2, double threshold) {
  if (fabs(f1-f2) > threshold) return false;
  return true;
}

#define APPROX_EQ(LHS, RHS, THRES)                                    \
    {                                                                \
      if (fabs(LHS - RHS) > THRES) {                                               \
        fprintf(stderr, "Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\n", #LHS, LHS, #RHS, RHS, THRES, \
                __FILE__, __LINE__, __PRETTY_FUNCTION__);               \
        abort();                                                        \
      }                                                                 \
    };
