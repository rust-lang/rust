#![warn(clippy::modulo_arithmetic)]
#![allow(clippy::no_effect, clippy::unnecessary_operation, clippy::modulo_one)]

fn main() {
    // Lint when both sides are const and of the opposite sign
    -1.6 % 2.1;
    //~^ ERROR: you are using modulo operator on constants with different signs: `-1.600 %
    //~| NOTE: double check for expected result especially when interoperating with differ
    1.6 % -2.1;
    //~^ ERROR: you are using modulo operator on constants with different signs: `1.600 %
    //~| NOTE: double check for expected result especially when interoperating with differ
    (1.1 - 2.3) % (1.1 + 2.3);
    //~^ ERROR: you are using modulo operator on constants with different signs: `-1.200 %
    //~| NOTE: double check for expected result especially when interoperating with differ
    (1.1 + 2.3) % (1.1 - 2.3);
    //~^ ERROR: you are using modulo operator on constants with different signs: `3.400 %
    //~| NOTE: double check for expected result especially when interoperating with differ

    // Lint on floating point numbers
    let a_f32: f32 = -1.6;
    let mut b_f32: f32 = 2.1;
    a_f32 % b_f32;
    //~^ ERROR: you are using modulo operator on types that might have different signs
    //~| NOTE: double check for expected result especially when interoperating with differ
    b_f32 % a_f32;
    //~^ ERROR: you are using modulo operator on types that might have different signs
    //~| NOTE: double check for expected result especially when interoperating with differ
    b_f32 %= a_f32;
    //~^ ERROR: you are using modulo operator on types that might have different signs
    //~| NOTE: double check for expected result especially when interoperating with differ

    let a_f64: f64 = -1.6;
    let mut b_f64: f64 = 2.1;
    a_f64 % b_f64;
    //~^ ERROR: you are using modulo operator on types that might have different signs
    //~| NOTE: double check for expected result especially when interoperating with differ
    b_f64 % a_f64;
    //~^ ERROR: you are using modulo operator on types that might have different signs
    //~| NOTE: double check for expected result especially when interoperating with differ
    b_f64 %= a_f64;
    //~^ ERROR: you are using modulo operator on types that might have different signs
    //~| NOTE: double check for expected result especially when interoperating with differ

    // No lint when both sides are const and of the same sign
    1.6 % 2.1;
    -1.6 % -2.1;
    (1.1 + 2.3) % (-1.1 + 2.3);
    (-1.1 - 2.3) % (1.1 - 2.3);
}
