
// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

extern crate collections;
use std::collections::BitVec;

fn bitv_test() {
    let mut v1 = box BitVec::from_elem(31, false);
    let v2 = box BitVec::from_elem(31, true);
    v1.union(&*v2);
}

pub fn main() {
    for _ in 0..10000 { bitv_test(); }
}
