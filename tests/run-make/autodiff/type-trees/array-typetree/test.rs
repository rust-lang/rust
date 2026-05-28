#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_test, Duplicated, Active)]
#[no_mangle]
fn test_array(arr: &[f64; 5]) -> f64 {
    arr[0] + arr[1] + arr[2] + arr[3] + arr[4]
}

fn main() {
    let arr = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut d_arr = [0.0; 5];
    let _result = d_test(&arr, &mut d_arr, 1.0);
}
