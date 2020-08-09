// Checks whether conditions in traits can be evaluated.
// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

trait IsZeroTrait<const IS_ZERO: bool>{}

impl IsZeroTrait<{0u8 == 0u8}> for () {}

impl IsZeroTrait<true> for ((),) {}

fn main() {}
