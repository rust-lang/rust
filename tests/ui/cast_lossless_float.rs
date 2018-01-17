#[warn(cast_lossless)]
#[allow(no_effect, unnecessary_operation)]
fn main() {
    // Test cast_lossless with casts to floating-point types
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
