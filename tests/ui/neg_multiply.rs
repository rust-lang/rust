// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::neg_multiply)]
#![allow(clippy::no_effect, clippy::unnecessary_operation)]

use std::ops::Mul;

struct X;

impl Mul<isize> for X {
    type Output = X;

    fn mul(self, _r: isize) -> Self {
        self
    }
}

impl Mul<X> for isize {
    type Output = X;

    fn mul(self, _r: X) -> X {
        X
    }
}

fn main() {
    let x = 0;

    x * -1;

    -1 * x;

    -1 * -1; // should be ok

    X * -1; // should be ok
    -1 * X; // should also be ok
}
