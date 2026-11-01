#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &u128) -> u128 {
    *x * *x
}

fn main() {
    let x: u128 = 7;
    let _ = callee(&x);
}
