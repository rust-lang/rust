// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
