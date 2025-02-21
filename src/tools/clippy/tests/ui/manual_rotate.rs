#![warn(clippy::manual_rotate)]
#![allow(unused)]
fn main() {
    let (x_u8, x_u16, x_u32, x_u64) = (1u8, 1u16, 1u32, 1u64);
    let (x_i8, x_i16, x_i32, x_i64) = (1i8, 1i16, 1i32, 1i64);
    let a_u32 = 1u32;
    // True positives
    let y_u8 = (x_u8 >> 3) | (x_u8 << 5);
    //~^ manual_rotate
    let y_u16 = (x_u16 >> 7) | (x_u16 << 9);
    //~^ manual_rotate
    let y_u32 = (x_u32 >> 8) | (x_u32 << 24);
    //~^ manual_rotate
    let y_u64 = (x_u64 >> 9) | (x_u64 << 55);
    //~^ manual_rotate
    let y_i8 = (x_i8 >> 3) | (x_i8 << 5);
    //~^ manual_rotate
    let y_i16 = (x_i16 >> 7) | (x_i16 << 9);
    //~^ manual_rotate
    let y_i32 = (x_i32 >> 8) | (x_i32 << 24);
    //~^ manual_rotate
    let y_i64 = (x_i64 >> 9) | (x_i64 << 55);
    //~^ manual_rotate
    // Plus also works instead of |
    let y_u32_plus = (x_u32 >> 8) + (x_u32 << 24);
    //~^ manual_rotate
    // Complex expression
    let y_u32_complex = ((x_u32 | 3256) >> 8) | ((x_u32 | 3256) << 24);
    //~^ manual_rotate
    let y_u64_as = (x_u32 as u64 >> 8) | ((x_u32 as u64) << 56);
    //~^ manual_rotate

    // False positives - can't be replaced with a rotation
    let y_u8_false = (x_u8 >> 6) | (x_u8 << 3);
    let y_u32_false = (x_u32 >> 8) | (x_u32 >> 24);
    let y_u64_false2 = (x_u64 >> 9) & (x_u64 << 55);
    // Variable mismatch
    let y_u32_wrong_vars = (x_u32 >> 8) | (a_u32 << 24);
    // Has side effects and therefore should not be matched
    let mut l = vec![12_u8, 34];
    let y = (l.pop().unwrap() << 3) + (l.pop().unwrap() >> 5);
}
