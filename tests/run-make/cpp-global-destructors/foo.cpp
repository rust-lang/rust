#include <stdint.h>

struct A {
    A() { v = 1234; }
    ~A() { v = 1; }
    uint32_t v;
};

A a;

extern "C" {
    uint32_t get() {
        return a.v;
    }
}
