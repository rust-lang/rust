// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core_intrinsics)]

use std::intrinsics::{fadd_fast, fsub_fast, fmul_fast, fdiv_fast, frem_fast};

#[inline(never)]
pub fn test_operations(a: f64, b: f64) {
    // make sure they all map to the correct operation
    unsafe {
        assert_eq!(fadd_fast(a, b), a + b);
        assert_eq!(fsub_fast(a, b), a - b);
        assert_eq!(fmul_fast(a, b), a * b);
        assert_eq!(fdiv_fast(a, b), a / b);
        assert_eq!(frem_fast(a, b), a % b);
    }
}

fn main() {
    test_operations(1., 2.);
    test_operations(10., 5.);
}
