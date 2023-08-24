#![allow(unused_parens)]
#![warn(clippy::verbose_bit_mask)]

fn main() {
    let x: i32 = 42;
    let _ = (x & 0b1111 == 0);
    //~^ ERROR: bit mask could be simplified with a call to `trailing_zeros`
    //~| NOTE: `-D clippy::verbose-bit-mask` implied by `-D warnings`
    let _ = x & 0b1_1111 == 0;
    //~^ ERROR: bit mask could be simplified with a call to `trailing_zeros`
    let _ = x & 0b1_1010 == 0; // do not lint
    let _ = x & 1 == 0; // do not lint
}
