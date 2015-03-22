// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for type inference failure around shifting. In this
// case, the iteration yields an int, but we hadn't run the full type
// propagation yet, and so we just saw a type variable, yielding an
// error.

// pretty-expanded FIXME #23616

#![feature(core)]

use std::u8;

trait IntoIterator {
    type Iter: Iterator;

    fn into_iter(self) -> Self::Iter;
}

impl<I> IntoIterator for I where I: Iterator {
    type Iter = I;

    fn into_iter(self) -> I {
        self
    }
}

fn desugared_for_loop_bad(byte: u8) -> u8 {
    let mut result = 0;
    let mut x = IntoIterator::into_iter(0..u8::BITS);
    let mut y = Iterator::next(&mut x);
    let mut z = y.unwrap();
    byte >> z;
    1
}

fn main() {}
