// ignore-tidy-file-linelength

_Complex float pass_simple_complex_float(_Complex float x) { return x; }
_Complex double pass_simple_complex_double(_Complex double x) { return x; }
_Complex long double pass_simple_complex_long_double(_Complex long double x) { return x; }

_Complex char pass_simple_complex_char(_Complex char x) { return x; }
_Complex short pass_simple_complex_short(_Complex short x) { return x; }
_Complex int pass_simple_complex_int(_Complex int x) { return x; }
_Complex long pass_simple_complex_long(_Complex long x) { return x; }
_Complex long long pass_simple_complex_long_long(_Complex long long x) { return x; }



_Complex float complex_float_align_int(int a0, _Complex float value) { return value; }
_Complex double complex_double_align_int(int a0, _Complex double value) { return value; }
_Complex long double complex_long_double_align_int(int a0, _Complex long double value) { return value; }

_Complex char complex_char_align_int(int a0, _Complex char value) { return value; }
_Complex short complex_short_align_int(int a0, _Complex short value) { return value; }
_Complex int complex_int_align_int(int a0, _Complex int value) { return value; }
_Complex long complex_long_align_int(int a0, _Complex long value) { return value; }
_Complex long long complex_long_long_align_int(int a0, _Complex long long value) { return value; }



_Complex float complex_float_align_float(float a0, _Complex float value) { return value; }
_Complex double complex_double_align_float(float a0, _Complex double value) { return value; }
_Complex long double complex_long_double_align_float(float a0, _Complex long double value) { return value; }

_Complex char complex_char_align_float(float a0, _Complex char value) { return value; }
_Complex short complex_short_align_float(float a0, _Complex short value) { return value; }
_Complex int complex_int_align_float(float a0, _Complex int value) { return value; }
_Complex long complex_long_align_float(float a0, _Complex long value) { return value; }
_Complex long long complex_long_long_align_float(float a0, _Complex long long value) { return value; }



int spill_trailing_complex_float_1(int a0, int a1, int a2, int a3, int a4, int a5, int a6, _Complex float value, int x) { return x; }

int spill_trailing_complex_double_1(int a0, int a1, int a2, int a3, int a4, int a5, _Complex double value, int x) { return x; }
int spill_trailing_complex_double_2(int a0, int a1, int a2, int a3, int a4, _Complex double value, int x, int y) { return y; }
int spill_trailing_complex_double_3(int a0, int a1, int a2, int a3, _Complex double value, int x, int y, int z) { return z; }

int spill_trailing_complex_long_double_1(int a0, _Complex long double value, int x) { return x; }



float spill_trailing_complex_float_1_float(float a0, float a1, float a2, float a3, float a4, float a5, float a6, _Complex float value, float x) { return x; }
float spill_trailing_complex_float_2_float(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7, _Complex float value, float x) { return x; }

float spill_trailing_complex_double_1_float(float a0, float a1, float a2, float a3, float a4, float a5, _Complex double value, float x) { return x; }
float spill_trailing_complex_double_2_float(float a0, float a1, float a2, float a3, float a4, float a5, float a6, _Complex double value, float x, float y) { return y; }
float spill_trailing_complex_double_3_float(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7, _Complex double value, float x, float y) { return y; }

float spill_trailing_complex_long_double_1_float(float a0, float a1, float a2, float a3, float a4, float a5, _Complex long double value, float x) { return x; }


_Complex float partial_complex_float(
    int a0, int a1, int a2, int a3,
    int a4, int a5, int a6,
    _Complex float value) {
  return value;
}

_Complex double partial_complex_double(
    int a0, int a1, int a2, int a3,
    int a4, int a5, int a6,
    _Complex double value) {
  return value;
}

_Complex long double partial_complex_long_double(
    int a0, int a1, int a2, int a3,
    int a4, int a5, int a6,
    _Complex long double value) {
  return value;
}

_Complex char partial_complex_char(
    int a0, int a1, int a2, int a3,
    int a4, int a5, int a6,
    _Complex char value) {
  return value;
}

_Complex short partial_complex_short(
    int a0, int a1, int a2, int a3,
    int a4, int a5, int a6,
    _Complex short value) {
  return value;
}

_Complex int partial_complex_int(
    int a0, int a1, int a2, int a3,
    int a4, int a5, int a6,
    _Complex int value) {
  return value;
}

_Complex long partial_complex_long(
    int a0, int a1, int a2, int a3,
    int a4, int a5, int a6,
    _Complex long value) {
  return value;
}

_Complex long long partial_complex_long_long(
    int a0, int a1, int a2, int a3,
    int a4, int a5, int a6,
    _Complex long long value) {
  return value;
}
