// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::util;
use std::num::Zero;

#[deriving(Zero)]
struct A;
#[deriving(Zero)]
struct B(int);
#[deriving(Zero)]
struct C(int, int);
#[deriving(Zero)]
struct D { a: int }
#[deriving(Zero)]
struct E { a: int, b: int }

#[deriving(Zero)]
struct Lots {
    c: Option<util::NonCopyable>,
    d: u8,
    e: char,
    f: float,
    g: (f32, char),
    h: ~[util::NonCopyable],
    i: @mut (int, int),
    j: bool,
    k: (),
}

fn main() {
    let lots: Lots = Zero::zero();
    assert!(lots.is_zero());
}
