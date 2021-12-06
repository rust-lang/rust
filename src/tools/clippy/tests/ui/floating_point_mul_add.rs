// run-rustfix
#![feature(const_fn_floating_point_arithmetic)]
#![warn(clippy::suboptimal_flops)]

/// Allow suboptimal_ops in constant context
pub const fn in_const_context() {
    let a: f64 = 1234.567;
    let b: f64 = 45.67834;
    let c: f64 = 0.0004;

    let _ = a * b + c;
    let _ = c + a * b;
}

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

    let _ = (a * a + b).sqrt();

    // Cases where the lint shouldn't be applied
    let _ = (a * a + b * b).sqrt();
}
