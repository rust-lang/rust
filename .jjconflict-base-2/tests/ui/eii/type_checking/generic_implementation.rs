//@ check-fail
// Check that type parameters on EIIs are properly rejected.
// Specifically a regression test for https://github.com/rust-lang/rust/issues/149983.
#![feature(extern_item_impls)]

#[eii]
fn foo();

#[foo]
fn foo_impl<T>() {}
//~^ ERROR `foo_impl` cannot have generic parameters other than lifetimes

fn main() {}
