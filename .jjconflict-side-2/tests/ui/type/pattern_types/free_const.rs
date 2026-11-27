//@ check-pass

#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

const START: u32 = 0;
const END: u32 = 10;

fn foo(_: pattern_type!(u32 is START..=END)) {}

fn main() {}
