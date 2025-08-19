#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &i32) -> i32 {
    *x * *x
}

fn main() {
    let x: i32 = 7;
    let _ = callee(&x);
}
