// run-rustfix

#![allow(clippy::no_effect, clippy::unnecessary_operation, dead_code)]
#![warn(clippy::cast_lossless)]

fn main() {
    // Test clippy::cast_lossless with casts to floating-point types
    1i8 as f32;
    1i8 as f64;
    1u8 as f32;
    1u8 as f64;
    1i16 as f32;
    1i16 as f64;
    1u16 as f32;
    1u16 as f64;
    1i32 as f64;
    1u32 as f64;
}

// The lint would suggest using `f64::from(input)` here but the `XX::from` function is not const,
// so we skip the lint if the expression is in a const fn.
// See #3656
const fn abc(input: f32) -> f64 {
    input as f64
}
