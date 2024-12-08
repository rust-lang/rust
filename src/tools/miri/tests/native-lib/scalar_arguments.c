#include <stdio.h>

// See comments in build_native_lib()
#define EXPORT __attribute__((visibility("default")))

EXPORT int add_one_int(int x) {
  return 2 + x;
}

EXPORT void printer(void) {
  printf("printing from C\n");
}

// function with many arguments, to test functionality when some args are stored
// on the stack
EXPORT int test_stack_spill(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, int k, int l) {
  return a+b+c+d+e+f+g+h+i+j+k+l;
}

EXPORT unsigned int get_unsigned_int(void) {
  return -10;
}

EXPORT short add_int16(short x) {
  return x + 3;
}

EXPORT long add_short_to_long(short x, long y) {
  return x + y;
}

// To test that functions not marked with EXPORT cannot be called by Miri.
int not_exported(void) {
  return 0;
}
