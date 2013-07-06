#include <stddef.h>

struct slice {
  char const *p;
  size_t len;
};

extern "C"
void test() {
  struct slice s = { .p = "hello",
                     .len = 5 };
}
