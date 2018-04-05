// ignore-license
#include <stdio.h>
#include <stdint.h>

typedef union TestUnion {
    uint64_t bits;
} TestUnion;

uint64_t give_back(TestUnion tu) {
    return tu.bits;
}
