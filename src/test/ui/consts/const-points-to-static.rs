// compile-flags: -Zunleash-the-miri-inside-of-you
// stderr-per-bitwidth

#![allow(dead_code)]

const TEST: &u8 = &MY_STATIC;
//~^ ERROR it is undefined behavior to use this value
//~| NOTE  encountered a reference pointing to a static variable
//~| NOTE undefined behavior
//~| NOTE the raw bytes of the constant

static MY_STATIC: u8 = 4;

fn main() {
}
