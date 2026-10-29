//@ check-pass

// if `X` were used instead of `x`, `X - 10` would result in a lint.
// This file should never produce a lint, no matter how the const
// propagator is improved.

#![deny(warnings)]

const X: u32 = 5;

fn main() {
    let x = X;
    if x > 10 {
        println!("{}", x - 10);
    } else {
        println!("{}", 10 - x);
    }
}
