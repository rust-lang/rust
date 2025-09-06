#![feature(autodiff, f128)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_test, Duplicated, Active)]
#[no_mangle]
fn test_f128(x: &f128) -> f128 {
    *x * *x
}

fn main() {
    let x = 2.0_f128;
    let mut dx = 0.0_f128;
    let _result = d_test(&x, &mut dx, 1.0);
}
