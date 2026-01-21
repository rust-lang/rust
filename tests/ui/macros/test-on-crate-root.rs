// Regression test for rust-lang/rust#114920
//
// Applying `#![test]` to the crate root used to ICE,
// when referring to the attribute with full path specifically.
#![core::prelude::v1::test]
//~^ ERROR inner macro attributes are unstable
//~| ERROR the `#[test]` attribute may only be used on a free function


fn main() {}
