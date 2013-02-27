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
#include "rust_task.h"
#include "rust_env.h"

extern struct type_desc str_body_tydesc;

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

inline void reserve_vec_exact_shared(rust_task* task, rust_vec_box** vpp,
                                     size_t size) {
    rust_opaque_box** ovpp = (rust_opaque_box**)vpp;
    if (size > (*vpp)->body.alloc) {
        *vpp = (rust_vec_box*)task->boxed.realloc(
            *ovpp, size + sizeof(rust_vec));
        (*vpp)->body.alloc = size;
    }
}

inline void reserve_vec_exact(rust_vec_box** vpp,
                              size_t size) {
    if (size > (*vpp)->body.alloc) {
        rust_exchange_alloc exchange_alloc;
        *vpp = (rust_vec_box*)exchange_alloc
            .realloc(*vpp, size + sizeof(rust_vec_box));
        (*vpp)->body.alloc = size;
    }
}

typedef rust_vec_box rust_str;

inline rust_str *
make_str(rust_kernel* kernel, const char* c, size_t strlen,
         const char* name) {
    size_t str_fill = strlen + 1;
    size_t str_alloc = str_fill;
    rust_str *str = (rust_str *)
        kernel->malloc(vec_size<char>(str_fill), name);
    str->header.td = &str_body_tydesc;
    str->body.fill = str_fill;
    str->body.alloc = str_alloc;
    memcpy(&str->body.data, c, strlen);
    str->body.data[strlen] = '\0';
    return str;
}

inline rust_vec_box *
make_str_vec(rust_kernel* kernel, size_t nstrs, char **strs) {
    rust_vec_box *v = (rust_vec_box *)
        kernel->malloc(vec_size<rust_vec_box*>(nstrs),
                       "str vec interior");
    // FIXME: should have a real td (Issue #2639)
    v->header.td = NULL;
    v->body.fill = v->body.alloc = sizeof(rust_vec_box*) * nstrs;
    for (size_t i = 0; i < nstrs; ++i) {
        rust_str *str = make_str(kernel, strs[i],
                                 strlen(strs[i]),
                                 "str");
        ((rust_str**)&v->body.data)[i] = str;
    }
    return v;
}

inline size_t get_box_size(size_t body_size, size_t body_align) {
    size_t header_size = sizeof(rust_opaque_box);
    // FIXME (#2699): This alignment calculation is suspicious. Is it right?
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
