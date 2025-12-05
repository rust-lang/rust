//! Check that pattern types patterns must be of the type of the base type

#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

const BAD_NESTING4: pattern_type!(u8 is 'a'..='a') = todo!();
//~^ ERROR: mismatched types
//~| ERROR: mismatched types

const BAD_NESTING5: pattern_type!(char is 1..=1) = todo!();
//~^ ERROR: mismatched types
//~| ERROR: mismatched types

fn main() {}
