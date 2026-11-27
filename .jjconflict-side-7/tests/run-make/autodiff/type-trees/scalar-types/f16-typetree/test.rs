#![feature(autodiff, f16)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_test, Duplicated, Active)]
#[no_mangle]
fn test_f16(x: &f16) -> f16 {
    *x * *x
}

fn main() {
    let x = 2.0_f16;
    let mut dx = 0.0_f16;
    let _result = d_test(&x, &mut dx, 1.0);
}
