//@ check-pass

#![allow(overflowing_literals)]

fn main() {
    let x: std::num::NonZero<i8> = -128;
    assert_eq!(x.get(), -128_i8);
}
