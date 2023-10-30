use autodiff::autodiff;

use ndarray::Array1;

#[autodiff(d_collect, Reverse, Active)]
fn collect(#[dup] x: &Array1<f32>) -> f32 {
    x[0]
}

fn main() {
    let a = Array1::zeros(19);
    let mut d_a = Array1::zeros(19);

    d_collect(&a, &mut d_a, 1.0);

    dbg!(&d_a);
}

#[cfg(test)]
mod tests {
    #[test]
    fn main() {
        super::main()
    }
}
