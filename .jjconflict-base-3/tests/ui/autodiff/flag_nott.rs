//@ compile-flags: -Zautodiff=Enable,NoTT
//@ needs-enzyme
//@ check-pass

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

// Test that NoTT flag is accepted and doesn't cause compilation errors
#[autodiff_reverse(d_square, Duplicated, Active)]
fn square(x: &f64) -> f64 {
    x * x
}

fn main() {
    let x = 2.0;
    let mut dx = 0.0;
    let result = d_square(&x, &mut dx, 1.0);
}
