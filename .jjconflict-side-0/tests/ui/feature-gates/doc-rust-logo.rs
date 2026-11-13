#![doc(rust_logo)]
//~^ ERROR this subset of the `doc` attribute is meant for internal use only
//! This is not an official rust crate

#[doc(rust_logo)]
//~^ WARN this attribute can only be applied at the crate level
fn main() {}
