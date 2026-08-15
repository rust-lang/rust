//@ check-fail
// Check that type parameters on EIIs are properly rejected.
// Specifically a regression test for https://github.com/rust-lang/rust/issues/149983.
#![feature(extern_item_impls)]

#[eii]
fn foo<T>() {}
//~^ ERROR externally implementable items may not have type parameters

fn main() {}
