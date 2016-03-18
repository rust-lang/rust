// ignore-license
#include <stddef.h>
#include <stdint.h>

size_t slice_len(uint8_t *data, size_t len) {
    return len;
}

uint8_t slice_elem(uint8_t *data, size_t len, size_t idx) {
    return data[idx];
}
