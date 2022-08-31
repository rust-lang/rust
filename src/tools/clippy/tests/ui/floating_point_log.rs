// run-rustfix
#![allow(dead_code, clippy::double_parens)]
#![warn(clippy::suboptimal_flops, clippy::imprecise_flops)]

const TWO: f32 = 2.0;
const E: f32 = std::f32::consts::E;

fn check_log_base() {
    let x = 1f32;
    let _ = x.log(2f32);
    let _ = x.log(10f32);
    let _ = x.log(std::f32::consts::E);
    let _ = x.log(TWO);
    let _ = x.log(E);
    let _ = (x as f32).log(2f32);

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
    let _ = (1.0 + x / 2.0).ln();
    let _ = (1.0 + x.powi(3)).ln();
    let _ = (1.0 + x.powi(3) / 2.0).ln();
    let _ = (1.0 + (std::f32::consts::E - 1.0)).ln();
    let _ = (x + 1.0).ln();
    let _ = (x.powi(3) + 1.0).ln();
    let _ = (x + 2.0 + 1.0).ln();
    let _ = (x / 2.0 + 1.0).ln();
    // Cases where the lint shouldn't be applied
    let _ = (1.0 + x + 2.0).ln();
    let _ = (x + 1.0 + 2.0).ln();
    let _ = (x + 1.0 / 2.0).ln();
    let _ = (1.0 + x - 2.0).ln();

    let x = 1f64;
    let _ = (1f64 + 2.).ln();
    let _ = (1f64 + 2.0).ln();
    let _ = (1.0 + x).ln();
    let _ = (1.0 + x / 2.0).ln();
    let _ = (1.0 + x.powi(3)).ln();
    let _ = (x + 1.0).ln();
    let _ = (x.powi(3) + 1.0).ln();
    let _ = (x + 2.0 + 1.0).ln();
    let _ = (x / 2.0 + 1.0).ln();
    // Cases where the lint shouldn't be applied
    let _ = (1.0 + x + 2.0).ln();
    let _ = (x + 1.0 + 2.0).ln();
    let _ = (x + 1.0 / 2.0).ln();
    let _ = (1.0 + x - 2.0).ln();
}

fn main() {}
