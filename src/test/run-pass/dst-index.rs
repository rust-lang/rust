// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that overloaded index expressions with DST result types
// work and don't ICE.

use std::ops::Index;
use std::fmt::Debug;

struct S;

impl Index<uint> for S {
    type Output = str;

    fn index<'a>(&'a self, _: &uint) -> &'a str {
        "hello"
    }
}

struct T;

impl Index<uint> for T {
    type Output = Debug + 'static;

    fn index<'a>(&'a self, idx: &uint) -> &'a (Debug + 'static) {
        static X: uint = 42;
        &X as &(Debug + 'static)
    }
}

fn main() {
    assert_eq!(&S[0], "hello");
    &T[0];
    // let x = &x as &Debug;
}
