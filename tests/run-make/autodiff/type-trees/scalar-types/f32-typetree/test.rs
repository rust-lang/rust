#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_test, Duplicated, Active)]
#[no_mangle]
fn test_f32(x: &f32) -> f32 {
    x * x
}

fn main() {
    let x = 2.0_f32;
    let mut dx = 0.0_f32;
    let _result = d_test(&x, &mut dx, 1.0);
}
