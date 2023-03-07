#![warn(clippy::modulo_arithmetic)]
#![allow(clippy::no_effect, clippy::unnecessary_operation, clippy::modulo_one)]

fn main() {
    // Lint when both sides are const and of the opposite sign
    -1.6 % 2.1;
    1.6 % -2.1;
    (1.1 - 2.3) % (1.1 + 2.3);
    (1.1 + 2.3) % (1.1 - 2.3);

    // Lint on floating point numbers
    let a_f32: f32 = -1.6;
    let mut b_f32: f32 = 2.1;
    a_f32 % b_f32;
    b_f32 % a_f32;
    b_f32 %= a_f32;

    let a_f64: f64 = -1.6;
    let mut b_f64: f64 = 2.1;
    a_f64 % b_f64;
    b_f64 % a_f64;
    b_f64 %= a_f64;

    // No lint when both sides are const and of the same sign
    1.6 % 2.1;
    -1.6 % -2.1;
    (1.1 + 2.3) % (-1.1 + 2.3);
    (-1.1 - 2.3) % (1.1 - 2.3);
}
