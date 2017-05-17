#![feature(plugin)]
#![plugin(clippy)]

#[warn(cmp_nan)]
#[allow(float_cmp, no_effect, unnecessary_operation)]
fn main() {
    let x = 5f32;
    x == std::f32::NAN;
    x != std::f32::NAN;
    x < std::f32::NAN;
    x > std::f32::NAN;
    x <= std::f32::NAN;
    x >= std::f32::NAN;

    let y = 0f64;
    y == std::f64::NAN;
    y != std::f64::NAN;
    y < std::f64::NAN;
    y > std::f64::NAN;
    y <= std::f64::NAN;
    y >= std::f64::NAN;
}
