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
// aux-build:struct_destructuring_cross_crate.rs

extern mod struct_destructuring_cross_crate;

pub fn main() {
    let x = struct_destructuring_cross_crate::S { x: 1, y: 2 };
    let struct_destructuring_cross_crate::S { x: a, y: b } = x;
    assert!(a == 1);
    assert!(b == 2);
}
