// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn test1() {
    let mut ints = [0, ..32];
    ints[0] += 1;
    assert_eq!(ints[0], 1);
}

fn test2() {
    let mut ints = [0, ..32];
    for vec::each_mut(ints) |i| { *i += 22; }
    for ints.each |i| { assert!(*i == 22); }
}

pub fn main() {
    test1();
    test2();
}
