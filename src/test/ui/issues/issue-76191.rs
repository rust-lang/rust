// Regression test for diagnostic issue #76191
#![allow(non_snake_case)]

use std::ops::RangeInclusive;
const RANGE: RangeInclusive<i32> = 0..=255;

fn main() {
    let n: i32 = 1;
    match n {
        RANGE => {}
        //~^ ERROR mismatched types
        _ => {}
    }
}
