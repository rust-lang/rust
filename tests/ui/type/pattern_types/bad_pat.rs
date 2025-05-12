#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

type NonNullU32_2 = pattern_type!(u32 is 1..=);
//~^ ERROR: inclusive range with no end
type Positive2 = pattern_type!(i32 is 0..=);
//~^ ERROR: inclusive range with no end
type Wild = pattern_type!(() is _);
//~^ ERROR: pattern not supported in pattern types

fn main() {}
