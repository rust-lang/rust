// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
// error-pattern:index out of bounds

use std::u64;

#[cfg(target_arch="x86")]
fn main() {
    let x = ~[1u,2u,3u];

    // This should cause a bounds-check failure, but may not if we do our
    // bounds checking by truncating the index value to the size of the
    // machine word, losing relevant bits of the index value.

    // This test is only meaningful on 32-bit hosts.

    let idx = u64::max_value & !(u64::max_value >> 1u);
    error!("ov3 idx = 0x%8.8x%8.8x",
           (idx >> 32) as uint,
           idx as uint);

    // This should fail.
    error!("ov3 0x%x",  x[idx]);
}

#[cfg(target_arch="x86_64")]
fn main() {
    // This version just fails anyways, for symmetry on 64-bit hosts.
    let x = ~[1u,2u,3u];
    error!("ov3 0x%x",  x[200]);
}
