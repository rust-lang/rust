// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast
// aux-build:cci_const.rs

extern crate cci_const;
static foo: &'static str = cci_const::foopy;
static a: uint = cci_const::uint_val;
static b: uint = cci_const::uint_expr + 5;

pub fn main() {
    assert_eq!(a, 12);
    let foo2 = a;
    assert_eq!(foo2, cci_const::uint_val);
    assert_eq!(b, cci_const::uint_expr + 5);
    assert_eq!(foo, cci_const::foopy);
}
