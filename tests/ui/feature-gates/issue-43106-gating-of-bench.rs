// The non-crate level cases are in issue-43106-gating-of-builtin-attrs.rs.
// See issue-12997-1.rs and issue-12997-2.rs to see how `#[bench]` is
// handled in "weird places" when `--test` is passed.

#![feature(custom_inner_attributes)]

#![bench                   = "4100"]
//~^ ERROR `bench` attribute cannot be used at crate level
fn main() {}
