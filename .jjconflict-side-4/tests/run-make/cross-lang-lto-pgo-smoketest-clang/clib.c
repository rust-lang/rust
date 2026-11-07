#include <stdint.h>

uint32_t c_always_inlined() {
    return 1234;
}

__attribute__((noinline)) uint32_t c_never_inlined() {
    return 12345;
}
