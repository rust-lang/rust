#![feature(core_pattern_type, core_pattern_types)]

use std::pat::pattern_type;

fn main() {
    let x: pattern_type!(i32 0..1);
    //~^ ERROR expected one of `!`, `(`, `+`, `::`, `<`, or `is`, found `0`
}
