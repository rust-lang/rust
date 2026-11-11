//@ compile-flags: --emit link
// The above flags (which are passed in `cargo build`) are required for
// the `arithmetic_overflow` lint to function.

#![deny(overflowing_literals, arithmetic_overflow)]

fn main() {
    let x: i8 = 128;
    //~^ ERROR literal out of range for `i8`
    let x: i8 = -128;
    let x: i8 = --128; //~ WARN use of a double negation
    //~^ ERROR this arithmetic operation will overflow
    let x: i8 = ---128; //~ WARN use of a double negation
    //~^ ERROR this arithmetic operation will overflow
    let x: i8 = ----128; //~ WARN use of a double negation
    //~^ ERROR this arithmetic operation will overflow
    let x: i8 = -----128; //~ WARN use of a double negation
    //~^ ERROR this arithmetic operation will overflow
    let x: i8 = ------128; //~ WARN use of a double negation
    //~^ ERROR this arithmetic operation will overflow
}
