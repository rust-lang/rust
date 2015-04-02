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
// can't be used as rvalues

use std::ops::Index;
use std::fmt::Debug;

#[derive(Copy, Clone)]
struct S;

impl Index<usize> for S {
    type Output = str;

    fn index(&self, _: usize) -> &str {
        "hello"
    }
}

#[derive(Copy, Clone)]
struct T;

impl Index<usize> for T {
    type Output = Debug + 'static;

    fn index<'a>(&'a self, idx: usize) -> &'a (Debug + 'static) {
        static x: usize = 42;
        &x
    }
}

fn main() {
    S[0];
    //~^ ERROR cannot move out of indexed content
    //~^^ ERROR E0161
    T[0];
    //~^ ERROR cannot move out of indexed content
    //~^^ ERROR E0161
}
