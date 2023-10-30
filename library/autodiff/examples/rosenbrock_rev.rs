use autodiff::autodiff;

#[autodiff(d_rosenbrock, Reverse, Active)]
fn rosenbrock(#[dup] x: &[f64; 2]) -> f64 {
    let mut res = 0.0;
    for i in 0..(x.len() - 1) {
        let a = x[i + 1] - x[i] * x[i];
        let b = x[i] - 1.0;
        res += 100.0 * a * a + b * b;
    }
    res
}

fn main() {
    let x = [3.14, 2.4];
    let output = rosenbrock(&x);
    println!("{output}");

    let mut df_dx = [0.0f64; 2];
    d_rosenbrock(&x, &mut df_dx, 1.0);

    // https://www.wolframalpha.com/input?i2d=true&i=x%3D3.14%3B+y%3D2.4%3B+D%5Brosenbrock+function%5C%2840%29x%5C%2844%29+y%5C%2841%29+%2Cy%5D
    assert!((df_dx[0] - 9373.54).abs() < 0.01);
    assert!((df_dx[1] - (-1491.92)).abs() < 0.01);
}

#[cfg(test)]
mod tests {
    #[test]
    fn main() {
        super::main()
    }
}
