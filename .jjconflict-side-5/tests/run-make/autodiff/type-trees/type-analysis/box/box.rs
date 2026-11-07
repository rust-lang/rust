#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &Box<f32>) -> f32 {
    **x * **x
}

fn main() {
    let x = Box::new(7.0f32);
    let mut df_dx = Box::new(0.0f32);
    let out = callee(&x);
    let out_ = d_square(&x, &mut df_dx, 1.0);
    assert_eq!(out, out_);
    assert_eq!(14.0, *df_dx);
}
