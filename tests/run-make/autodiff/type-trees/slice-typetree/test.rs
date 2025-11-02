#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_test, Duplicated, Active)]
#[no_mangle]
fn test_slice(slice: &[f64]) -> f64 {
    slice.iter().sum()
}

fn main() {
    let arr = [1.0, 2.0, 3.0, 4.0, 5.0];
    let slice = &arr[..];
    let mut d_slice = [0.0; 5];
    let _result = d_test(slice, &mut d_slice[..], 1.0);
}
