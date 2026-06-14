//! Test that #[inline(always)] can be applied to main function

//@ check-pass

#[inline(always)]
fn main() {}
