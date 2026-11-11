#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &u8) -> u8 {
    *x * *x
}

fn main() {
    let x: u8 = 7;
    let _ = callee(&x);
}
