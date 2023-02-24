#![feature(bench_black_box)]
use autodiff::autodiff;

type Matrix = Vec<Vec<f32>>;
type Vector = Vec<f32>;

#[autodiff(d_matvec, Reverse, Const)]
fn matvec(#[dup] mat: &Matrix, vec: &Vector, #[dup] out: &mut Vector) {
    for i in 0..mat.len() - 1{
        let mut int = 0.0;

        for j in 0..mat[0].len() - 1{
            for k in 0..vec.len() - 1{
                int += mat[i][j] * vec[k];
            }
        }

        out[i] = int;
    }
}

fn main() {
    let mat = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let mut d_mat = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    let mut inp = vec![1.0, 1.0];
    let mut out = vec![1.0, 1.0];
    let mut out_tang = vec![0.0, 1.0];

    d_matvec(&mat, &mut d_mat, &mut inp, &mut out, &mut out_tang);
    matvec(&mat, &inp, &mut out);

    dbg!(&out);
}
