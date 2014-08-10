
// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::hash::hash;

// testing multiple separate deriving attributes
#[deriving(PartialEq)]
#[deriving(Clone)]
#[deriving(Hash)]
struct Foo {
    bar: uint,
    baz: int
}

pub fn main() {
    let a = Foo {bar: 4, baz: -3};

    a == a;    // check for PartialEq impl w/o testing its correctness
    a.clone(); // check for Clone impl w/o testing its correctness
    hash(&a);  // check for Hash impl w/o testing its correctness
}
