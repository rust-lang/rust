#![feature(bench_black_box)]
use autodiff::autodiff;
use std::ptr;

#[autodiff(sin_vec, Reverse, Active)]
fn cos_vec(#[dup] x: &Vec<f32>) -> f32 {
    // uses enum internally and breaks
    let res = x.into_iter().collect::<Vec<&f32>>();

    *res[0]
}

fn main() {
    let x = vec![1.0, 1.0, 1.0];
    let mut d_x = vec![0.0; 3];

    sin_vec(&x, &mut d_x, 1.0);

    dbg!(&d_x, &x);
}
