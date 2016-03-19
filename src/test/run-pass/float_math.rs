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

fn main() {
    // make sure they all map to the correct operation
    unsafe {
        assert_eq!(fadd_fast(1., 2.), 1. + 2.);
        assert_eq!(fsub_fast(1., 2.), 1. - 2.);
        assert_eq!(fmul_fast(2., 3.), 2. * 3.);
        assert_eq!(fdiv_fast(10., 5.), 10. / 5.);
        assert_eq!(frem_fast(10., 5.), 10. % 5.);
    }
}
