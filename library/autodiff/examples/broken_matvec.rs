use autodiff::autodiff;

type Matrix = Vec<Vec<f32>>;
type Vector = Vec<f32>;

#[autodiff(d_matvec, Forward, Const)]
fn matvec(#[dup] mat: &Matrix, vec: &Vector, #[dup] out: &mut Vector) {
    for i in 0..mat.len() - 1 {
        for j in 0..mat[0].len() - 1 {
            out[i] += mat[i][j] * vec[j];
        }
    }
}

fn main() {
    let mat = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let mut d_mat = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    let inp = vec![1.0, 1.0];
    let mut out = vec![0.0, 0.0];
    let mut out_tang = vec![0.0, 1.0];

    //matvec(&mat, &inp, &mut out);
    d_matvec(&mat, &mut d_mat, &inp, &mut out, &mut out_tang);

    dbg!(&out);
}

#[cfg(test)]
mod tests {
    #[test]
    fn main() {
        super::main()
    }
}
