// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test
// error-pattern:index out of bounds

use std::u64;

#[cfg(target_arch="x86")]
fn main() {
    let x = vec!(1_usize,2_usize,3_usize);

    // This should cause a bounds-check panic, but may not if we do our
    // bounds checking by truncating the index value to the size of the
    // machine word, losing relevant bits of the index value.

    // This test is only meaningful on 32-bit hosts.

    let idx = u64::MAX & !(u64::MAX >> 1_usize);
    println!("ov3 idx = 0x%8.8x%8.8x",
           (idx >> 32) as uint,
           idx as uint);

    // This should panic.
    println!("ov3 0x%x",  x[idx]);
}

#[cfg(any(target_arch="x86_64", target_arch = "aarch64"))]
fn main() {
    // This version just panics anyways, for symmetry on 64-bit hosts.
    let x = vec!(1_usize,2_usize,3_usize);
    error!("ov3 0x%x",  x[200]);
}
