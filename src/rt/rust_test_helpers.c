// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Helper functions used only in tests

#include "rust_globals.h"

// These functions are used in the unit tests for C ABI calls.

uint32_t
rust_dbg_extern_identity_u32(uint32_t u) {
    return u;
}

uint64_t
rust_dbg_extern_identity_u64(uint64_t u) {
    return u;
}

double
rust_dbg_extern_identity_double(double u) {
    return u;
}

char
rust_dbg_extern_identity_u8(char u) {
    return u;
}

typedef void *(*dbg_callback)(void*);

void *
rust_dbg_call(dbg_callback cb, void *data) {
    return cb(data);
}

void rust_dbg_do_nothing() { }

struct TwoU8s {
    uint8_t one;
    uint8_t two;
};

struct TwoU8s
rust_dbg_extern_return_TwoU8s() {
    struct TwoU8s s;
    s.one = 10;
    s.two = 20;
    return s;
}

struct TwoU8s
rust_dbg_extern_identity_TwoU8s(struct TwoU8s u) {
    return u;
}

struct TwoU16s {
    uint16_t one;
    uint16_t two;
};

struct TwoU16s
rust_dbg_extern_return_TwoU16s() {
    struct TwoU16s s;
    s.one = 10;
    s.two = 20;
    return s;
}

struct TwoU16s
rust_dbg_extern_identity_TwoU16s(struct TwoU16s u) {
    return u;
}

struct TwoU32s {
    uint32_t one;
    uint32_t two;
};

struct TwoU32s
rust_dbg_extern_return_TwoU32s() {
    struct TwoU32s s;
    s.one = 10;
    s.two = 20;
    return s;
}

struct TwoU32s
rust_dbg_extern_identity_TwoU32s(struct TwoU32s u) {
    return u;
}

struct TwoU64s {
    uint64_t one;
    uint64_t two;
};

struct TwoU64s
rust_dbg_extern_return_TwoU64s() {
    struct TwoU64s s;
    s.one = 10;
    s.two = 20;
    return s;
}

struct TwoU64s
rust_dbg_extern_identity_TwoU64s(struct TwoU64s u) {
    return u;
}

struct TwoDoubles {
    double one;
    double two;
};

struct TwoDoubles
rust_dbg_extern_identity_TwoDoubles(struct TwoDoubles u) {
    return u;
}

intptr_t
rust_get_test_int() {
  return 1;
}

/* Debug helpers strictly to verify ABI conformance.
 *
 * FIXME (#2665): move these into a testcase when the testsuite
 * understands how to have explicit C files included.
 */

struct quad {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint64_t d;
};

struct floats {
    double a;
    uint8_t b;
    double c;
};

struct quad
rust_dbg_abi_1(struct quad q) {
    struct quad qq = { q.c + 1,
                       q.d - 1,
                       q.a + 1,
                       q.b - 1 };
    return qq;
}

struct floats
rust_dbg_abi_2(struct floats f) {
    struct floats ff = { f.c + 1.0,
                         0xff,
                         f.a - 1.0 };
    return ff;
}

int
rust_dbg_static_mut;

int rust_dbg_static_mut = 3;

void
rust_dbg_static_mut_check_four() {
    assert(rust_dbg_static_mut == 4);
}
