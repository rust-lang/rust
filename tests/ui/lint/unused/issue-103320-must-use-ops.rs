//@ check-pass

#![warn(unused_must_use)]
#![feature(never_type)]

use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use std::ops::Div;
use std::ops::Rem;

fn main() {
    let x = 2_u32;
    (x.add(4), x.sub(4), x.mul(4), x.div(4), x.rem(4));

    x.add(4); //~ WARN unused return value of `add` that must be used

    x.sub(4); //~ WARN unused return value of `sub` that must be used

    x.mul(4); //~ WARN unused return value of `mul` that must be used

    x.div(4); //~ WARN unused return value of `div` that must be used

    x.rem(4); //~ WARN unused return value of `rem` that must be used

    println!("{}", x);
}
