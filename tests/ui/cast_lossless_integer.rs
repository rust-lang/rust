// run-rustfix

#[warn(clippy::cast_lossless)]
#[allow(clippy::no_effect, clippy::unnecessary_operation)]
fn main() {
    // Test clippy::cast_lossless with casts to integer types
    1i8 as i16;
    1i8 as i32;
    1i8 as i64;
    1u8 as i16;
    1u8 as i32;
    1u8 as i64;
    1u8 as u16;
    1u8 as u32;
    1u8 as u64;
    1i16 as i32;
    1i16 as i64;
    1u16 as i32;
    1u16 as i64;
    1u16 as u32;
    1u16 as u64;
    1i32 as i64;
    1u32 as i64;
    1u32 as u64;
}
