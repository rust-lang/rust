#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_test, Duplicated, Active)]
#[no_mangle]
fn test_i32(x: &i32) -> i32 {
    x * x
}

fn main() {
    let x = 5_i32;
    let mut dx = 0_i32;
    let _result = d_test(&x, &mut dx, 1);
}
