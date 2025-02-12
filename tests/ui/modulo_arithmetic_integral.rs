#![warn(clippy::modulo_arithmetic)]
#![allow(clippy::no_effect, clippy::unnecessary_operation, clippy::modulo_one)]

fn main() {
    // Lint on signed integral numbers
    let a = -1;
    let mut b = 2;
    a % b;
    //~^ modulo_arithmetic

    b % a;
    //~^ modulo_arithmetic

    b %= a;
    //~^ modulo_arithmetic

    let a_i8: i8 = 1;
    let mut b_i8: i8 = 2;
    a_i8 % b_i8;
    //~^ modulo_arithmetic

    b_i8 %= a_i8;
    //~^ modulo_arithmetic

    let a_i16: i16 = 1;
    let mut b_i16: i16 = 2;
    a_i16 % b_i16;
    //~^ modulo_arithmetic

    b_i16 %= a_i16;
    //~^ modulo_arithmetic

    let a_i32: i32 = 1;
    let mut b_i32: i32 = 2;
    a_i32 % b_i32;
    //~^ modulo_arithmetic

    b_i32 %= a_i32;
    //~^ modulo_arithmetic

    let a_i64: i64 = 1;
    let mut b_i64: i64 = 2;
    a_i64 % b_i64;
    //~^ modulo_arithmetic

    b_i64 %= a_i64;
    //~^ modulo_arithmetic

    let a_i128: i128 = 1;
    let mut b_i128: i128 = 2;
    a_i128 % b_i128;
    //~^ modulo_arithmetic

    b_i128 %= a_i128;
    //~^ modulo_arithmetic

    let a_isize: isize = 1;
    let mut b_isize: isize = 2;
    a_isize % b_isize;
    //~^ modulo_arithmetic

    b_isize %= a_isize;
    //~^ modulo_arithmetic

    let a = 1;
    let mut b = 2;
    a % b;
    //~^ modulo_arithmetic

    b %= a;
    //~^ modulo_arithmetic

    // No lint on unsigned integral value
    let a_u8: u8 = 17;
    let b_u8: u8 = 3;
    a_u8 % b_u8;
    let mut a_u8: u8 = 1;
    a_u8 %= 2;

    let a_u16: u16 = 17;
    let b_u16: u16 = 3;
    a_u16 % b_u16;
    let mut a_u16: u16 = 1;
    a_u16 %= 2;

    let a_u32: u32 = 17;
    let b_u32: u32 = 3;
    a_u32 % b_u32;
    let mut a_u32: u32 = 1;
    a_u32 %= 2;

    let a_u64: u64 = 17;
    let b_u64: u64 = 3;
    a_u64 % b_u64;
    let mut a_u64: u64 = 1;
    a_u64 %= 2;

    let a_u128: u128 = 17;
    let b_u128: u128 = 3;
    a_u128 % b_u128;
    let mut a_u128: u128 = 1;
    a_u128 %= 2;

    let a_usize: usize = 17;
    let b_usize: usize = 3;
    a_usize % b_usize;
    let mut a_usize: usize = 1;
    a_usize %= 2;

    // No lint when comparing to zero
    let a = -1;
    let mut b = 2;
    let c = a % b == 0;
    let c = 0 == a % b;
    let c = a % b != 0;
    let c = 0 != a % b;
}
