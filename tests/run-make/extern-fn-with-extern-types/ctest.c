#include <stdio.h>
#include <stdint.h>

typedef struct data {
    uint32_t magic;
} data;

data* data_create(uint32_t magic) {
    static data d;
    d.magic = magic;
    return &d;
}

uint32_t data_get(data* p) {
    return p->magic;
}
