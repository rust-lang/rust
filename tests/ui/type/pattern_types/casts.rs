//! Check that pattern types can be cast where they can be coerced

#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

fn identity(x: pattern_type!(u32 is 1..)) -> pattern_type!(u32 is 1..) {
    x as pattern_type!(u32 is 1..)
}

fn main() {
    let x: pattern_type!(u32 is 1..) = 3;
    let y = x as u32; //~ ERROR `(u32) is 1..` as `u32`
    let z = x as u64; //~ ERROR `(u32) is 1..` as `u64`
}

fn bar() {
    let x = 4 as pattern_type!(u32 is 1..);
}

fn foo() -> u32 {
    5 as pattern_type!(u32 is 1..)
    //~^ ERROR: mismatched types
}

#[rustfmt::skip]
fn arms(b: bool) -> u32 {
    if b {
        24
    } else {
        6 as pattern_type!(u32 is 1..)
        //~^ ERROR: mismatched types
    }
}

#[rustfmt::skip]
fn arms2(b: bool) -> u32 {
    if b {
        7 as pattern_type!(u32 is 1..)
        //~^ ERROR: mismatched types
    } else {
        24
    }
}
