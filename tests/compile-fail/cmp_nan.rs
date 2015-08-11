#![feature(plugin)]
#![plugin(clippy)]

#[deny(cmp_nan)]
#[allow(float_cmp)]
fn main() {
    let x = 5f32;
    x == std::f32::NAN; //~ERROR
    x != std::f32::NAN; //~ERROR
    x < std::f32::NAN; //~ERROR
    x > std::f32::NAN; //~ERROR
    x <= std::f32::NAN; //~ERROR
    x >= std::f32::NAN; //~ERROR

    let y = 0f64;
    y == std::f64::NAN; //~ERROR
    y != std::f64::NAN; //~ERROR
    y < std::f64::NAN; //~ERROR
    y > std::f64::NAN; //~ERROR
    y <= std::f64::NAN; //~ERROR
    y >= std::f64::NAN; //~ERROR
}
