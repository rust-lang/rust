//@ check-pass

#![crate_type = "lib"]
#![warn(clippy::imprecise_flops)]
#![warn(clippy::suboptimal_flops)]
#![no_std]

// The following should not lint, as the suggested methods `{f16,f32,f64,f128}.mul_add()`
// and `{f16,f32,f64,f128}::abs()` are not available in no_std

pub fn mul_add() {
    let a: f64 = 1234.567;
    let b: f64 = 45.67834;
    let c: f64 = 0.0004;
    let _ = a * b + c;
}

fn fake_abs1(num: f64) -> f64 {
    if num >= 0.0 { num } else { -num }
}

pub fn main(_argc: isize, _argv: *const *const u8) -> isize {
    0
}
