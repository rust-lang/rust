// compile-flags: -Zunleash-the-miri-inside-of-you
// run-pass

#![allow(dead_code)]

const TEST: u8 = MY_STATIC;
//~^ skipping const checks

static MY_STATIC: u8 = 4;

fn main() {
}
