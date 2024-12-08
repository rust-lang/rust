#ifdef _WIN32
__declspec(dllexport)
#endif

// An external function, defined in Rust.
extern void panic_if_greater_than_10(unsigned x);

unsigned add_small_numbers(unsigned a, unsigned b) {
    unsigned c = a + b;
    panic_if_greater_than_10(c);
    return c;
}
