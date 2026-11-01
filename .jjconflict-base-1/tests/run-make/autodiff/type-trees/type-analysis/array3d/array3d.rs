#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &[[[f32; 2]; 2]; 2]) -> f32 {
    let mut sum = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                sum += x[i][j][k] * x[i][j][k];
            }
        }
    }
    sum
}

fn main() {
    let x = [[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let mut df_dx = [[[0.0f32; 2]; 2]; 2];
    let out = callee(&x);
    let out_ = d_square(&x, &mut df_dx, 1.0);
    assert_eq!(out, out_);
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                assert_eq!(df_dx[i][j][k], 2.0 * x[i][j][k]);
            }
        }
    }
}
