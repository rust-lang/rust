// check-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

trait Trait {}

fn f<const N: usize>(_: impl Trait) {}

fn main() {}
