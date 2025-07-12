#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &usize) -> usize {
    *x * *x
}

fn main() {
    let x: usize = 7;
    let _ = callee(&x);
}
