#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &u16) -> u16 {
    *x * *x
}

fn main() {
    let x: u16 = 7;
    let _ = callee(&x);
}
