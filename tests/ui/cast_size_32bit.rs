//@ignore-64bit
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
    //~^ ERROR: casting `isize` to `i8` may truncate the value
    let x0 = 1isize;
    let x1 = 1usize;
    x0 as f64;
    //~^ ERROR: casting `isize` to `f64` causes a loss of precision on targets with 64-bit
    //~| NOTE: `-D clippy::cast-precision-loss` implied by `-D warnings`
    x1 as f64;
    //~^ ERROR: casting `usize` to `f64` causes a loss of precision on targets with 64-bit
    x0 as f32;
    //~^ ERROR: casting `isize` to `f32` causes a loss of precision (`isize` is 32 or 64 b
    x1 as f32;
    //~^ ERROR: casting `usize` to `f32` causes a loss of precision (`usize` is 32 or 64 b
    1isize as i32;
    //~^ ERROR: casting `isize` to `i32` may truncate the value on targets with 64-bit wid
    1isize as u32;
    //~^ ERROR: casting `isize` to `u32` may truncate the value on targets with 64-bit wid
    1usize as u32;
    //~^ ERROR: casting `usize` to `u32` may truncate the value on targets with 64-bit wid
    1usize as i32;
    //~^ ERROR: casting `usize` to `i32` may truncate the value on targets with 64-bit wid
    //~| ERROR: casting `usize` to `i32` may wrap around the value on targets with 32-bit
    //~| NOTE: `-D clippy::cast-possible-wrap` implied by `-D warnings`
    // Casting to *size
    1i64 as isize;
    //~^ ERROR: casting `i64` to `isize` may truncate the value on targets with 32-bit wid
    1i64 as usize;
    //~^ ERROR: casting `i64` to `usize` may truncate the value on targets with 32-bit wid
    1u64 as isize;
    //~^ ERROR: casting `u64` to `isize` may truncate the value on targets with 32-bit wid
    //~| ERROR: casting `u64` to `isize` may wrap around the value on targets with 64-bit
    1u64 as usize;
    //~^ ERROR: casting `u64` to `usize` may truncate the value on targets with 32-bit wid
    1u32 as isize;
    //~^ ERROR: casting `u32` to `isize` may wrap around the value on targets with 32-bit
    1u32 as usize; // Should not trigger any lint
    1i32 as isize; // Neither should this
    1i32 as usize;
    // Big integer literal to float
    999_999_999 as f32;
    //~^ ERROR: casting `i32` to `f32` causes a loss of precision (`i32` is 32 bits wide,
    3_999_999_999usize as f64;
    //~^ ERROR: casting integer literal to `f64` is unnecessary
    //~| NOTE: `-D clippy::unnecessary-cast` implied by `-D warnings`
}
