// Regression test for #92230.
//
//@ check-pass
//@ compile-flags: -Znext-solver

#![feature(const_trait_impl)]

pub const trait Super {}
pub const trait Sub: Super {}

const impl<A> Super for &A where A: [const] Super {}
const impl<A> Sub for &A where A: [const] Sub {}

fn main() {}
