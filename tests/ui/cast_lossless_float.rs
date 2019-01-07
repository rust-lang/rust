// run-rustfix

#[warn(clippy::cast_lossless)]
#[allow(clippy::no_effect, clippy::unnecessary_operation)]
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
