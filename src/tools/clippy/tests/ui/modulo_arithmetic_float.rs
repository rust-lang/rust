#![feature(f128)]
#![feature(f16)]
#![warn(clippy::modulo_arithmetic)]
#![allow(clippy::no_effect, clippy::unnecessary_operation, clippy::modulo_one)]

fn main() {
    // Lint when both sides are const and of the opposite sign
    -1.6 % 2.1;
    //~^ modulo_arithmetic

    1.6 % -2.1;
    //~^ modulo_arithmetic

    (1.1 - 2.3) % (1.1 + 2.3);
    //~^ modulo_arithmetic

    (1.1 + 2.3) % (1.1 - 2.3);
    //~^ modulo_arithmetic

    // Lint on floating point numbers
    let a_f16: f16 = -1.6;
    let mut b_f16: f16 = 2.1;
    a_f16 % b_f16;
    //~^ modulo_arithmetic

    b_f16 % a_f16;
    //~^ modulo_arithmetic

    b_f16 %= a_f16;
    //~^ modulo_arithmetic

    // Lint on floating point numbers
    let a_f32: f32 = -1.6;
    let mut b_f32: f32 = 2.1;
    a_f32 % b_f32;
    //~^ modulo_arithmetic

    b_f32 % a_f32;
    //~^ modulo_arithmetic

    b_f32 %= a_f32;
    //~^ modulo_arithmetic

    let a_f64: f64 = -1.6;
    let mut b_f64: f64 = 2.1;
    a_f64 % b_f64;
    //~^ modulo_arithmetic

    b_f64 % a_f64;
    //~^ modulo_arithmetic

    b_f64 %= a_f64;
    //~^ modulo_arithmetic

    let a_f128: f128 = -1.6;
    let mut b_f128: f128 = 2.1;
    a_f128 % b_f128;
    //~^ modulo_arithmetic

    b_f128 % a_f128;
    //~^ modulo_arithmetic

    b_f128 %= a_f128;
    //~^ modulo_arithmetic

    // No lint when both sides are const and of the same sign
    1.6 % 2.1;
    -1.6 % -2.1;
    (1.1 + 2.3) % (-1.1 + 2.3);
    (-1.1 - 2.3) % (1.1 - 2.3);
}
