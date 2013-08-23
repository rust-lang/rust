// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#ifndef RUST_UTIL_H
#define RUST_UTIL_H

#include <limits.h>
#include "rust_type.h"

// Inline fn used regularly elsewhere.

// Rounds |size| to the nearest |alignment|. Invariant: |alignment| is a power
// of two.
template<typename T>
static inline T
align_to(T size, size_t alignment) {
    assert(alignment);
    T x = (T)(((uintptr_t)size + alignment - 1) & ~(alignment - 1));
    return x;
}

// Interior vectors (rust-user-code level).

struct
rust_vec
{
    size_t fill;    // in bytes; if zero, heapified
    size_t alloc;   // in bytes
    uint8_t data[0];
};

struct
rust_vec_box
{
    rust_opaque_box header;
    rust_vec body;
};

template <typename T>
inline size_t vec_size(size_t elems) {
    return sizeof(rust_vec_box) + sizeof(T) * elems;
}

template <typename T>
inline T *
vec_data(rust_vec *v) {
    return reinterpret_cast<T*>(v->data);
}

typedef rust_vec rust_str;

inline size_t get_box_size(size_t body_size, size_t body_align) {
    size_t header_size = sizeof(rust_opaque_box);
    size_t total_size = align_to(header_size, body_align) + body_size;
    return total_size;
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//

#endif
