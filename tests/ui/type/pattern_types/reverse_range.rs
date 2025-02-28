//! Check that the range start must be smaller than the range end
#![feature(pattern_types, const_trait_impl, pattern_type_range_trait)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

const NONE: pattern_type!(u8 is 1..0) = unsafe { std::mem::transmute(3_u8) };
//~^ NOTE: attempt to compute `0_u8 - 1_u8`, which would overflow
//~| ERROR: evaluation of constant value failed

fn main() {}
