// ignore-fast

// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(globs)];
#[allow(dead_assignment)];

extern crate extra;
use std::vec::*;

pub fn main() {
    let mut v = from_elem(0u, 0);
    v = append(v, [4, 2]);
    assert_eq!(from_fn(2, |i| 2*(i+1)), ~[2, 4]);
}
