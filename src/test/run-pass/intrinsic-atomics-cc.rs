// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast - check-fast doesn't understand aux-build
// aux-build:cci_intrinsic.rs

extern mod cci_intrinsic;
use cci_intrinsic::atomic_xchg;

pub fn main() {
    unsafe {
        let mut x = 1;
        atomic_xchg(&mut x, 5);
        assert!(x == 5);
    }
}
