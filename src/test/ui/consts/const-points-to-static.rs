// compile-flags: -Zunleash-the-miri-inside-of-you

#![allow(dead_code)]

const TEST: &u8 = &MY_STATIC;
//~^ skipping const checks
//~| it is undefined behavior to use this value

static MY_STATIC: u8 = 4;

fn main() {
}
