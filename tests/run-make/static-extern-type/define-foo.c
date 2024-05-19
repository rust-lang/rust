#include <stdint.h>

struct Foo {
    uint8_t x;
};

struct Foo FOO = { 42 };

uint8_t bar(const struct Foo* foo) {
    return foo->x;
}
