
// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate collections;
use collections::Bitv;

fn bitv_test() {
    let mut v1 = box Bitv::new(31, false);
    let v2 = box Bitv::new(31, true);
    v1.union(v2);
}

pub fn main() {
    for _ in range(0, 10000) { bitv_test(); }
}
