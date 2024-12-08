// Checks that `impl Trait<{anon_const}> for Type` evaluates successfully.
//@ check-pass
//@ revisions: full min

#![cfg_attr(full, feature(generic_const_exprs))]
#![cfg_attr(full, allow(incomplete_features))]

trait IsZeroTrait<const IS_ZERO: bool>{}

impl IsZeroTrait<{0u8 == 0u8}> for () {}

impl IsZeroTrait<true> for ((),) {}

fn main() {}
