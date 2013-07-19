#include <stdlib.h>

size_t foo(size_t x) {
    return x * x;
}

extern "C"
void test() {
    size_t x = foo(10);
}
