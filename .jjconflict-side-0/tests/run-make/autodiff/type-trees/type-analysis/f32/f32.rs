#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &f32) -> f32 {
    *x * *x
}

fn main() {
    let x: f32 = 7.0;
    let mut df_dx: f32 = 0.0;
    let out = callee(&x);
    let out_ = d_square(&x, &mut df_dx, 1.0);

    assert_eq!(out, out_);
    assert_eq!(14.0, df_dx);
}
