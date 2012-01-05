// This is a helper C program for generating required math constants
//
// Should only be required when porting to a different target architecture
// (or c compiler/libmath)
//
// Call with <rust machine type of c_float> <rust machine type of c_double>
// and ensure that libcore/cmath.rs complies to the output
//
// Requires a printf that supports "%a" specifiers
//

#include <float.h>
#include <math.h>
#include <stdio.h>

// must match core::ctypes

#define C_FLT(x) (float)x
#define C_DBL(x) (double)x

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "%s <ctypes::c_float> <ctypes::c_double>\n", argv[0]);
    return 1;
  }
  char* c_flt = argv[1];
  char* c_dbl = argv[2];

  printf("mod c_float_math_consts {\n");
  printf("    const pi: c_float = %a_%s;\n", C_FLT(M_PI), c_flt);
  printf("    const div_1_pi: c_float = %a_%s;\n", C_FLT(M_1_PI), c_flt);
  printf("    const div_2_pi: c_float = %a_%s;\n", C_FLT(M_2_PI), c_flt);
  printf("    const div_pi_2: c_float = %a_%s;\n", C_FLT(M_PI_2), c_flt);
  printf("    const div_pi_4: c_float = %a_%s;\n", C_FLT(M_PI_4), c_flt);
  printf("    const div_2_sqrtpi: c_float = %a_%s;\n",
         C_FLT(M_2_SQRTPI), c_flt);
  printf("    const e: c_float = %a_%s;\n", C_FLT(M_E), c_flt);
  printf("    const log2_e: c_float = %a_%s;\n", C_FLT(M_LOG2E), c_flt);
  printf("    const log10_e: c_float = %a_%s;\n", C_FLT(M_LOG10E), c_flt);
  printf("    const ln_2: c_float = %a_%s;\n",  C_FLT(M_LN2), c_flt);
  printf("    const ln_10: c_float = %a_%s;\n",  C_FLT(M_LN10), c_flt);
  printf("    const sqrt2: c_float = %a_%s;\n",  C_FLT(M_SQRT2), c_flt);
  printf("    const div_1_sqrt2: c_float = %a_%s;\n",
         C_FLT(M_SQRT1_2), c_flt);
  printf("}\n\n");

  printf("mod c_double_math_consts {\n");
  printf("    const pi: c_double = %a_%s;\n", C_DBL(M_PI), c_dbl);
  printf("    const div_1_pi: c_double = %a_%s;\n", C_DBL(M_1_PI), c_dbl);
  printf("    const div_2_pi: c_double = %a_%s;\n", C_DBL(M_2_PI), c_dbl);
  printf("    const div_pi_2: c_double = %a_%s;\n", C_DBL(M_PI_2), c_dbl);
  printf("    const div_pi_4: c_double = %a_%s;\n", C_DBL(M_PI_4), c_dbl);
  printf("    const div_2_sqrtpi: c_double = %a_%s;\n",
         C_DBL(M_2_SQRTPI), c_dbl);
  printf("    const e: c_double = %a_%s;\n", C_DBL(M_E), c_dbl);
  printf("    const log2_e: c_double = %a_%s;\n", C_DBL(M_LOG2E), c_dbl);
  printf("    const log10_e: c_double = %a_%s;\n", C_DBL(M_LOG10E), c_dbl);
  printf("    const ln_2: c_double = %a_%s;\n", C_DBL(M_LN2), c_dbl);
  printf("    const ln_10: c_double = %a_%s;\n", C_DBL(M_LN10), c_dbl);
  printf("    const sqrt2: c_double = %a_%s;\n", C_DBL(M_SQRT2), c_dbl);
  printf("    const div_1_sqrt2: c_double = %a_%s;\n",
         C_DBL(M_SQRT1_2), c_dbl);
  printf("}\n\n");

  printf("mod c_float_targ_consts {\n");
  printf("    const radix: uint = %uu;\n", FLT_RADIX);
  printf("    const mantissa_digits: uint = %uu;\n", FLT_MANT_DIG);
  printf("    const digits: uint = %uu;\n", FLT_DIG);
  printf("    const min_exp: int = %i;\n", FLT_MIN_EXP);
  printf("    const max_exp: int = %i;\n", FLT_MAX_EXP);
  printf("    const min_10_exp: int = %i;\n", FLT_MIN_10_EXP);
  printf("    const max_10_exp: int = %i;\n", FLT_MAX_10_EXP);
  printf("    const min_value: c_float = %a_%s;\n", C_FLT(FLT_MIN), c_flt);
  printf("    const max_value: c_float = %a_%s;\n", C_FLT(FLT_MAX), c_flt);
  printf("    const epsilon: c_float = %a_%s;\n", C_FLT(FLT_EPSILON), c_flt);
  printf("}\n\n");

  printf("mod c_double_targ_consts {\n");
  printf("    const radix: uint = %uu;\n", FLT_RADIX);
  printf("    const mantissa_digits: uint = %uu;\n", DBL_MANT_DIG);
  printf("    const digits: uint = %uu;\n", DBL_DIG);
  printf("    const min_exp: int = %i;\n", DBL_MIN_EXP);
  printf("    const max_exp: int = %i;\n", DBL_MAX_EXP);
  printf("    const min_10_exp: int = %i;\n", DBL_MIN_10_EXP);
  printf("    const max_10_exp: int = %i;\n", DBL_MAX_10_EXP);
  printf("    const min_value: c_double = %a_%s;\n", C_DBL(DBL_MIN), c_dbl);
  printf("    const max_value: c_double = %a_%s;\n", C_DBL(DBL_MAX), c_dbl);
  printf("    const epsilon: c_double = %a_%s;\n", C_DBL(DBL_EPSILON), c_dbl);
  printf("}\n");

  return 0;
}
