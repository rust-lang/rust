#![warn(clippy::neg_multiply)]
#![allow(clippy::no_effect, clippy::unnecessary_operation, clippy::precedence)]
#![allow(unused)]

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
    //~^ neg_multiply

    -1 * x;
    //~^ neg_multiply

    100 + x * -1;
    //~^ neg_multiply

    (100 + x) * -1;
    //~^ neg_multiply

    -1 * 17;
    //~^ neg_multiply

    0xcafe | 0xff00 * -1;
    //~^ neg_multiply

    3_usize as i32 * -1;
    //~^ neg_multiply
    (3_usize as i32) * -1;
    //~^ neg_multiply

    -1 * -1; // should be ok

    X * -1; // should be ok
    -1 * X; // should also be ok
}
