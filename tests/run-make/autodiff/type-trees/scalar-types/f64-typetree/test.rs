#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_test, Duplicated, Active)]
#[no_mangle]
fn test_f64(x: &f64) -> f64 {
    x * x
}

fn main() {
    let x = 2.0_f64;
    let mut dx = 0.0_f64;
    let _result = d_test(&x, &mut dx, 1.0);
}
