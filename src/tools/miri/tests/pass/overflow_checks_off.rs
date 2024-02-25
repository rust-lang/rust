//@compile-flags: -C overflow-checks=off

// Check that we correctly implement the intended behavior of these operators
// when they are not being overflow-checked at runtime.

// FIXME: if we call the functions in `std::ops`, we still get the panics.
// Miri does not implement the codegen-time hack that backs `#[rustc_inherit_overflow_checks]`.
// use std::ops::*;

// Disable _compile-time_ overflow linting
// so that we can test runtime overflow checks
#![allow(arithmetic_overflow)]

fn main() {
    assert_eq!(-{ -0x80i8 }, -0x80);

    assert_eq!(0xffu8 + 1, 0_u8);
    assert_eq!(0u8 - 1, 0xff_u8);
    assert_eq!(0xffu8 * 2, 0xfe_u8);
    assert_eq!(1u8 << 9, 2_u8);
    assert_eq!(2u8 >> 9, 1_u8);
}
