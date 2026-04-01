#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &i16) -> i16 {
    *x * *x
}

fn main() {
    let x: i16 = 7;
    let _ = callee(&x);
}
