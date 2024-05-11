#include <stdio.h>

int add_one_int(int x) {
  return 2 + x;
}

void printer() {
  printf("printing from C\n");
}

// function with many arguments, to test functionality when some args are stored
// on the stack
int test_stack_spill(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, int k, int l) {
  return a+b+c+d+e+f+g+h+i+j+k+l;
}

unsigned int get_unsigned_int() {
  return -10;
}

short add_int16(short x) {
  return x + 3;
}

long add_short_to_long(short x, long y) {
  return x + y;
}
