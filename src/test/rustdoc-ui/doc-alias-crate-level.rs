#![doc(alias = "crate-level-not-working")] //~ ERROR

#[doc(alias = "shouldn't work!")] //~ ERROR
pub fn foo() {}
