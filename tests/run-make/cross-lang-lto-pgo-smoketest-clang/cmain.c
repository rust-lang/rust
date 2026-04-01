#include <stdint.h>

// A trivial function defined in Rust, returning a constant value. This should
// always be inlined.
uint32_t rust_always_inlined();


uint32_t rust_never_inlined();

int main(int argc, char** argv) {
    return (rust_never_inlined() + rust_always_inlined()) * 0;
}
