#![feature(pattern_types)]
#![feature(pattern_type_macro)]
#![feature(inline_const_pat)]
//@ check-pass

use std::pat::pattern_type;

fn bar(x: pattern_type!(u32 is 0..=const{ 5 + 5 })) {}

fn main() {}
