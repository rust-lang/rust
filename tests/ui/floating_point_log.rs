#![allow(dead_code)]
#![warn(clippy::floating_point_improvements)]

const TWO: f32 = 2.0;
const E: f32 = std::f32::consts::E;

fn check_log_base() {
    let x = 1f32;
    let _ = x.log(2f32);
    let _ = x.log(10f32);
    let _ = x.log(std::f32::consts::E);
    let _ = x.log(TWO);
    let _ = x.log(E);

    let x = 1f64;
    let _ = x.log(2f64);
    let _ = x.log(10f64);
    let _ = x.log(std::f64::consts::E);
}

fn check_ln1p() {
    let x = 1f32;
    let _ = (1f32 + 2.).ln();
    let _ = (1f32 + 2.0).ln();
    let _ = (1.0 + x).ln();
    let _ = (1.0 + x * 2.0).ln();
    let _ = (1.0 + x.powi(2)).ln();
    let _ = (1.0 + x.powi(2) * 2.0).ln();
    let _ = (1.0 + (std::f32::consts::E - 1.0)).ln();
    let _ = (x + 1.0).ln();
    let _ = (x.powi(2) + 1.0).ln();
    let _ = (x + 2.0 + 1.0).ln();
    let _ = (x * 2.0 + 1.0).ln();
    // Cases where the lint shouldn't be applied
    let _ = (1.0 + x + 2.0).ln();
    let _ = (x + 1.0 + 2.0).ln();
    let _ = (x + 1.0 * 2.0).ln();
    let _ = (1.0 + x - 2.0).ln();

    let x = 1f64;
    let _ = (1f64 + 2.).ln();
    let _ = (1f64 + 2.0).ln();
    let _ = (1.0 + x).ln();
    let _ = (1.0 + x * 2.0).ln();
    let _ = (1.0 + x.powi(2)).ln();
    let _ = (x + 1.0).ln();
    let _ = (x.powi(2) + 1.0).ln();
    let _ = (x + 2.0 + 1.0).ln();
    let _ = (x * 2.0 + 1.0).ln();
    // Cases where the lint shouldn't be applied
    let _ = (1.0 + x + 2.0).ln();
    let _ = (x + 1.0 + 2.0).ln();
    let _ = (x + 1.0 * 2.0).ln();
    let _ = (1.0 + x - 2.0).ln();
}

fn check_log_division() {
    let x = 3f32;
    let y = 2f32;
    let b = 4f32;

    let _ = x.log2() / y.log2();
    let _ = x.log10() / y.log10();
    let _ = x.ln() / y.ln();
    let _ = x.log(4.0) / y.log(4.0);
    let _ = x.log(b) / y.log(b);
    let _ = x.log(b) / y.log(x);
    let _ = x.log(b) / 2f32.log(b);

    let x = 3f64;
    let y = 2f64;
    let b = 4f64;

    let _ = x.log2() / y.log2();
    let _ = x.log10() / y.log10();
    let _ = x.ln() / y.ln();
    let _ = x.log(4.0) / y.log(4.0);
    let _ = x.log(b) / y.log(b);
    let _ = x.log(b) / y.log(x);
    let _ = x.log(b) / 2f64.log(b);
}

fn main() {}
