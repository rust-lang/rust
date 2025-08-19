#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &u32) -> u32 {
    *x * *x
}

fn main() {
    let x: u32 = 7;
    let _ = callee(&x);
}
