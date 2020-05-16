// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

trait IsZeroTrait<const IS_ZERO: bool>{}

impl IsZeroTrait<{0u8 == 0u8}> for () {}

impl IsZeroTrait<true> for ((),) {}

fn main() {}
