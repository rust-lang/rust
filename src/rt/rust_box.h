/* Rust box representation. */

#ifndef RUST_BOX_H
#define RUST_BOX_H

#include "rust_internal.h"
#include <stdint.h>

struct rust_box {
    RUST_REFCOUNTED(rust_box)
    type_desc *tydesc;
    rust_box *gc_next;
    rust_box *gc_prev;
    uint8_t data[0];
};

#endif

