#![warn(clippy::modulo_arithmetic)]
#![allow(
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::modulo_one,
    clippy::identity_op
)]

fn main() {
    // Lint when both sides are const and of the opposite sign
    -1 % 2;
    //~^ ERROR: you are using modulo operator on constants with different signs: `-1 % 2`
    //~| NOTE: double check for expected result especially when interoperating with differ
    1 % -2;
    //~^ ERROR: you are using modulo operator on constants with different signs: `1 % -2`
    //~| NOTE: double check for expected result especially when interoperating with differ
    (1 - 2) % (1 + 2);
    //~^ ERROR: you are using modulo operator on constants with different signs: `-1 % 3`
    //~| NOTE: double check for expected result especially when interoperating with differ
    (1 + 2) % (1 - 2);
    //~^ ERROR: you are using modulo operator on constants with different signs: `3 % -1`
    //~| NOTE: double check for expected result especially when interoperating with differ
    35 * (7 - 4 * 2) % (-500 * -600);
    //~^ ERROR: you are using modulo operator on constants with different signs: `-35 % 30
    //~| NOTE: double check for expected result especially when interoperating with differ

    -1i8 % 2i8;
    //~^ ERROR: you are using modulo operator on constants with different signs: `-1 % 2`
    //~| NOTE: double check for expected result especially when interoperating with differ
    1i8 % -2i8;
    //~^ ERROR: you are using modulo operator on constants with different signs: `1 % -2`
    //~| NOTE: double check for expected result especially when interoperating with differ
    -1i16 % 2i16;
    //~^ ERROR: you are using modulo operator on constants with different signs: `-1 % 2`
    //~| NOTE: double check for expected result especially when interoperating with differ
    1i16 % -2i16;
    //~^ ERROR: you are using modulo operator on constants with different signs: `1 % -2`
    //~| NOTE: double check for expected result especially when interoperating with differ
    -1i32 % 2i32;
    //~^ ERROR: you are using modulo operator on constants with different signs: `-1 % 2`
    //~| NOTE: double check for expected result especially when interoperating with differ
    1i32 % -2i32;
    //~^ ERROR: you are using modulo operator on constants with different signs: `1 % -2`
    //~| NOTE: double check for expected result especially when interoperating with differ
    -1i64 % 2i64;
    //~^ ERROR: you are using modulo operator on constants with different signs: `-1 % 2`
    //~| NOTE: double check for expected result especially when interoperating with differ
    1i64 % -2i64;
    //~^ ERROR: you are using modulo operator on constants with different signs: `1 % -2`
    //~| NOTE: double check for expected result especially when interoperating with differ
    -1i128 % 2i128;
    //~^ ERROR: you are using modulo operator on constants with different signs: `-1 % 2`
    //~| NOTE: double check for expected result especially when interoperating with differ
    1i128 % -2i128;
    //~^ ERROR: you are using modulo operator on constants with different signs: `1 % -2`
    //~| NOTE: double check for expected result especially when interoperating with differ
    -1isize % 2isize;
    //~^ ERROR: you are using modulo operator on constants with different signs: `-1 % 2`
    //~| NOTE: double check for expected result especially when interoperating with differ
    1isize % -2isize;
    //~^ ERROR: you are using modulo operator on constants with different signs: `1 % -2`
    //~| NOTE: double check for expected result especially when interoperating with differ

    // No lint when both sides are const and of the same sign
    1 % 2;
    -1 % -2;
    (1 + 2) % (-1 + 2);
    (-1 - 2) % (1 - 2);

    1u8 % 2u8;
    1u16 % 2u16;
    1u32 % 2u32;
    1u64 % 2u64;
    1u128 % 2u128;
    1usize % 2usize;
}
