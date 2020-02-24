// run-rustfix
#![warn(clippy::suboptimal_flops)]

fn main() {
    let a: f64 = 1234.567;
    let b: f64 = 45.67834;
    let c: f64 = 0.0004;
    let d: f64 = 0.0001;

    let _ = a * b + c;
    let _ = c + a * b;
    let _ = a + 2.0 * 4.0;
    let _ = a + 2. * 4.;

    let _ = (a * b) + c;
    let _ = c + (a * b);
    let _ = a * b * c + d;

    let _ = a.mul_add(b, c) * a.mul_add(b, c) + a.mul_add(b, c) + c;
    let _ = 1234.567_f64 * 45.67834_f64 + 0.0004_f64;
}
