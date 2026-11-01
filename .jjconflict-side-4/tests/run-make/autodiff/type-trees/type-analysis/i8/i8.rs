#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &i8) -> i8 {
    *x * *x
}

fn main() {
    let x: i8 = 7;
    let _ = callee(&x);
}
