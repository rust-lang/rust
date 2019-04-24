#[warn(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
#[allow(clippy::no_effect, clippy::unnecessary_operation)]
fn main() {
    // Test clippy::cast_precision_loss
    let x0 = 1i32;
    x0 as f32;
    let x1 = 1i64;
    x1 as f32;
    x1 as f64;
    let x2 = 1u32;
    x2 as f32;
    let x3 = 1u64;
    x3 as f32;
    x3 as f64;
    // Test clippy::cast_possible_truncation
    1f32 as i32;
    1f32 as u32;
    1f64 as f32;
    1i32 as i8;
    1i32 as u8;
    1f64 as isize;
    1f64 as usize;
    // Test clippy::cast_possible_wrap
    1u8 as i8;
    1u16 as i16;
    1u32 as i32;
    1u64 as i64;
    1usize as isize;
    // Test clippy::cast_sign_loss
    1i32 as u32;
    -1i32 as u32;
    1isize as usize;
    -1isize as usize;
    0i8 as u8;
    i8::max_value() as u8;
    i16::max_value() as u16;
    i32::max_value() as u32;
    i64::max_value() as u64;
    i128::max_value() as u128;
    // Extra checks for *size
    // Test cast_unnecessary
    1i32 as i32;
    1f32 as f32;
    false as bool;
    &1i32 as &i32;
    // macro version
    macro_rules! foo {
        ($a:ident, $b:ident) => {
            pub fn $a() -> $b {
                1 as $b
            }
        };
    }
    foo!(a, i32);
    foo!(b, f32);
    foo!(c, f64);

    // casting integer literal to float is unnecessary
    100 as f32;
    100 as f64;
    100_i32 as f64;
    // Should not trigger
    #[rustfmt::skip]
    let v = vec!(1);
    &v as &[i32];
    1.0 as f64;
    1 as u64;
}
