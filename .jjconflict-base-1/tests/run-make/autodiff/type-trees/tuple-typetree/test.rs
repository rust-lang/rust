#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_test, Duplicated, Active)]
#[no_mangle]
fn test_tuple(tuple: &(f64, f64, f64)) -> f64 {
    tuple.0 + tuple.1 * 2.0 + tuple.2 * 3.0
}

fn main() {
    let tuple = (1.0, 2.0, 3.0);
    let mut d_tuple = (0.0, 0.0, 0.0);
    let _result = d_test(&tuple, &mut d_tuple, 1.0);
}
