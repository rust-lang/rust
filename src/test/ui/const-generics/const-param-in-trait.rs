// Check that const parameters are permitted in traits.
// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]


trait Trait<const T: u8> {}

fn main() {}
