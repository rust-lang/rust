// https://github.com/rust-lang/rust/issues/76191
// Regression test for diagnostic issue #76191
#![allow(non_snake_case)]

use std::ops::RangeInclusive;

const RANGE: RangeInclusive<i32> = 0..=255;

const RANGE2: RangeInclusive<i32> = panic!();
//~^ ERROR evaluation panicked: explicit panic

fn main() {
    let n: i32 = 1;
    match n {
        RANGE => {}
        //~^ ERROR mismatched types
        RANGE2 => {}
        //~^ ERROR mismatched types
        _ => {}
    }
}
