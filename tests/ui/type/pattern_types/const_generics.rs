#![feature(pattern_types)]
#![feature(core_pattern_types)]
#![feature(core_pattern_type)]

use std::pat::pattern_type;

trait Foo {}

impl<const START: u32, const END: u32> Foo for pattern_type!(u32 is START..=END) {}
//~^ ERROR: range patterns must have constant range start and end
//~| ERROR: range patterns must have constant range start and end

fn main() {}
