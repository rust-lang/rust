//! Test that #[inline(always)] can be applied to main function

//@ run-pass

#[inline(always)]
fn main() {}

//@ reference: attributes.codegen.inline.allowed-positions
