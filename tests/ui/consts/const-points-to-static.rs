// compile-flags: -Zunleash-the-miri-inside-of-you
// stderr-per-bitwidth

#![allow(dead_code)]

const TEST: &u8 = &MY_STATIC;
//~^ ERROR it is undefined behavior to use this value
//~| encountered a reference pointing to a static variable

static MY_STATIC: u8 = 4;

fn main() {
}
