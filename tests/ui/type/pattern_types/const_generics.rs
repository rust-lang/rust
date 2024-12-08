//@ check-pass

#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

trait Foo {}

impl<const START: u32, const END: u32> Foo for pattern_type!(u32 is START..=END) {}

fn main() {}
