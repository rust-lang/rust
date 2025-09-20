#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn square(x: &f64) -> f64 {
    x * x
}

fn main() {
    let x = 2.0;
    let mut dx = 0.0;
    let _result = d_square(&x, &mut dx, 1.0);
}
