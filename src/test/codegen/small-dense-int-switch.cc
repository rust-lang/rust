#include <stdlib.h>

extern "C"
size_t test(size_t x, size_t y) {
  switch (x) {
  case 1: return y;
  case 2: return y*2;
  case 4: return y*3;
  default: return 11;
  }
}
