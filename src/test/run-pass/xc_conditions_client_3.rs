// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
// aux-build:xc_conditions_3.rs

extern mod xc_conditions_3;
use xcc = xc_conditions_3;

pub fn main() {
    assert_eq!(xcc::guard(a, 1), 40);
}

pub fn a() -> int {
    assert_eq!(xcc::oops::cond.raise(7), 7);
    xcc::guard(b, 2)
}

pub fn b() -> int {
    assert_eq!(xcc::oops::cond.raise(8), 16);
    xcc::guard(c, 3)
}

pub fn c() -> int {
    assert_eq!(xcc::oops::cond.raise(9), 27);
    xcc::guard(d, 4)
}

pub fn d() -> int {
    xcc::oops::cond.raise(10)
}
