#![feature(plugin)]

#![plugin(clippy)]
#![deny(neg_multiply)]
#![allow(no_effect, unnecessary_operation)]

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
