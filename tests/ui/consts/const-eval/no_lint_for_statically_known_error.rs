// if `X` were used instead of `x`, `X - 10` would result in a lint.
// This file should not produce a lint.
// This is a bug in the const propagator.

#![deny(warnings)]

const X: u32 = 5;

fn main() {
    let x = X;
    if x > 10 {
        println!("{}", x - 10);
        //~^ ERROR: this arithmetic operation will overflow
    } else {
        println!("{}", 10 - x);
    }
}
