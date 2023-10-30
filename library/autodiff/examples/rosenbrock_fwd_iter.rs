use autodiff::autodiff;

#[autodiff(d_rosenbrock, Forward, DuplicatedNoNeed)]
fn rosenbrock(#[dup] x: &[f64; 2]) -> f64 {
    (0..x.len() - 1)
        .map(|i| {
            let (a, b) = (x[i + 1] - x[i] * x[i], x[i] - 1.0);
            100.0 * a * a + b * b
        })
        .sum()
}

fn main() {
    let x = [3.14f64, 2.4];
    let output = rosenbrock(&x);
    println!("{output}");

    let df_dx = d_rosenbrock(&x, &[1.0, 0.0]);
    let df_dy = d_rosenbrock(&x, &[0.0, 1.0]);

    dbg!(&df_dx, &df_dy);

    // https://www.wolframalpha.com/input?i2d=true&i=x%3D3.14%3B+y%3D2.4%3B+D%5Brosenbrock+function%5C%2840%29x%5C%2844%29+y%5C%2841%29+%2Cy%5D
    assert!((df_dx - 9373.54).abs() < 0.1);
    assert!((df_dy - (-1491.92)).abs() < 0.1);
}

#[cfg(test)]
mod tests {
    #[test]
    fn main() {
        super::main()
    }
}
