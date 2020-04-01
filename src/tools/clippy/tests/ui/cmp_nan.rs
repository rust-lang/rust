const NAN_F32: f32 = std::f32::NAN;
const NAN_F64: f64 = std::f64::NAN;

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
    x == NAN_F32;
    x != NAN_F32;
    x < NAN_F32;
    x > NAN_F32;
    x <= NAN_F32;
    x >= NAN_F32;

    let y = 0f64;
    y == std::f64::NAN;
    y != std::f64::NAN;
    y < std::f64::NAN;
    y > std::f64::NAN;
    y <= std::f64::NAN;
    y >= std::f64::NAN;
    y == NAN_F64;
    y != NAN_F64;
    y < NAN_F64;
    y > NAN_F64;
    y <= NAN_F64;
    y >= NAN_F64;
}
