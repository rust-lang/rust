// Multiple custom crate-level attributes, both inert and active.

// check-pass
// aux-crate:test_macros=test-macros.rs

#![feature(custom_inner_attributes)]
#![feature(prelude_import)]

#![test_macros::identity_attr]
#![rustfmt::skip]
#![test_macros::identity_attr]
#![rustfmt::skip]

fn main() {}
