//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

#![feature(generic_const_items)]

const FAIL<'a, T>: &'a T = todo!();
//~^ ERROR: unable to satisfy constraints because it cannot be proven
fn main() {}
