#include <stddef.h>
#include <stdint.h>

struct ByteSlice {
        uint8_t *data;
        size_t len;
};

size_t slice_len(struct ByteSlice bs) {
        return bs.len;
}

uint8_t slice_elem(struct ByteSlice bs, size_t idx) {
        return bs.data[idx];
}
