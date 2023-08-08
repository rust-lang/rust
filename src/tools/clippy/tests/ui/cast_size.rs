//@ignore-32bit
#[warn(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless
)]
#[allow(clippy::no_effect, clippy::unnecessary_operation)]
fn main() {
    // Casting from *size
    1isize as i8;
    let x0 = 1isize;
    let x1 = 1usize;
    x0 as f64;
    x1 as f64;
    x0 as f32;
    x1 as f32;
    1isize as i32;
    1isize as u32;
    1usize as u32;
    1usize as i32;
    // Casting to *size
    1i64 as isize;
    1i64 as usize;
    1u64 as isize;
    1u64 as usize;
    1u32 as isize;
    1u32 as usize; // Should not trigger any lint
    1i32 as isize; // Neither should this
    1i32 as usize;
    // Big integer literal to float
    999_999_999 as f32;
    9_999_999_999_999_999usize as f64;
}
