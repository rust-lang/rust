#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &i128) -> i128 {
    *x * *x
}

fn main() {
    let x: i128 = 7;
    let _ = callee(&x);
}
