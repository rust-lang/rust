#![feature(bench_black_box)]
use autodiff::autodiff;

#[autodiff(d_sum, Reverse, Active)]
fn sum(#[dup] x: &Vec<Vec<f32>>) -> f32 {
    x.into_iter().map(|x| x.into_iter().map(|x| x.sqrt())).flatten().sum()
}

fn main() {
    let a = vec![vec![1.0, 2.0, 4.0, 8.0]];
    let mut b = vec![vec![0.0, 0.0, 0.0, 0.0]];

    d_sum(&a, &mut b, 1.0);

    dbg!(&b);
}
