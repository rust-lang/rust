#[warn(clippy::cmp_nan)]
#[allow(clippy::float_cmp, clippy::no_effect, clippy::unnecessary_operation)]
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
