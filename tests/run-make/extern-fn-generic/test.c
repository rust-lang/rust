#include <stdint.h>

typedef struct TestStruct {
        uint8_t x;
        int32_t y;
} TestStruct;

typedef int callback(TestStruct s);

uint32_t call(callback *c) {
        TestStruct s;
        s.x = 'a';
        s.y = 3;

        return c(s);
}
