#![feature(pattern_type_macro)]

use std::pat::pattern_type;

fn main() {
    let x: pattern_type!(i32 0..1);
    //~^ ERROR expected one of `!`, `(`, `+`, `::`, `<`, or `is`, found `0`
}
