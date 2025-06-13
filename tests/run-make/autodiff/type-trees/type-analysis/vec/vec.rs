#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(arg: &std::vec::Vec<f32>) -> f32 {
    arg.iter().sum()
}

fn main() {
    let v = vec![1.0f32, 2.0, 3.0];
    let _ = callee(&v);
}
