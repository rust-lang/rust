// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #41849.

use std::ops::Mul;

const C: usize = 1;
const CAPACITY: usize = 1 * C;

struct A<X> {
    f: [X; CAPACITY],
}

struct B<T> {
    f: T,
}

impl<T> Mul for B<T> {
    type Output = Self;
    fn mul(self, _rhs: B<T>) -> Self::Output {
        self
    }
}

impl<T> Mul<usize> for B<T> {
    type Output = Self;
    fn mul(self, _rhs: usize) -> Self::Output {
        self
    }
}

fn main() {
    let a = A { f: [1] };
    let _ = B { f: a };
}
