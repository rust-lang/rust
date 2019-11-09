#include <stdlib.h> 
#include <stdbool.h>
#include <math.h> 

static bool approx_fp_equality_float(float f1, float f2, double threshold) {
  if (fabs(f1-f2) > threshold) return false;
  return true;
}

static bool approx_fp_equality_double(double f1, double f2, double threshold) {
  if (fabs(f1-f2) > threshold) return false;
  return true;
}
