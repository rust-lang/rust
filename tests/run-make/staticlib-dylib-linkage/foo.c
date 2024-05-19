#include <assert.h>

extern void foo();
extern unsigned bar(unsigned a, unsigned b);

int main() {
  foo();
  assert(bar(1, 2) == 3);
  return 0;
}
