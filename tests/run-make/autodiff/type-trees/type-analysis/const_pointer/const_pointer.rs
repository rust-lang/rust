#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: *const f32) -> f32 {
    unsafe { *x * *x }
}

fn main() {
    let x: f32 = 7.0;
    let out = callee(&x as *const f32);
    let mut df_dx: f32 = 0.0;
    let out_ = d_square(&x as *const f32, &mut df_dx, 1.0);
    assert_eq!(out, out_);
    assert_eq!(14.0, df_dx);
}
