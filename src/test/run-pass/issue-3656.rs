// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast Can't redeclare malloc with wrong signature because bugs
// Issue #3656
// Incorrect struct size computation in the FFI, because of not taking
// the alignment of elements into account.

use core::libc::*;

struct KEYGEN {
    hash_algorithm: [c_uint, ..2],
    count: uint32_t,
    salt: *c_void,
    salt_size: uint32_t,
}

extern {
    // Bogus signature, just need to test if it compiles.
    pub fn malloc(data: KEYGEN);
}

pub fn main() {
}
