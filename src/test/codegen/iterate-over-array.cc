#include <stdlib.h>
#include <assert.h>

struct slice {
  int const *p;
  size_t len;
};

extern "C"
size_t test(slice s) {
  size_t y = 0;
  for (int i = 0; i < s.len; ++i) {
	assert(i < s.len);
	y += s.p[i];
  }
  return y;
}
