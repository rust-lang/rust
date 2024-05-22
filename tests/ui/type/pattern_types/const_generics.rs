//@ check-pass

#![feature(pattern_types)]
#![feature(core_pattern_types)]
#![feature(core_pattern_type)]

use std::pat::pattern_type;

trait Foo {}

impl<const START: u32, const END: u32> Foo for pattern_type!(u32 is START..=END) {}

fn main() {}
