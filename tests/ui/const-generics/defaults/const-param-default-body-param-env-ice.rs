//! Regression test for <https://github.com/rust-lang/rust/issues/148096>.

impl<const N: usize = { || [0; X] }, X> dyn PartialEq<X> {}
//~^ ERROR generic parameters with a default must be trailing
//~| ERROR generic parameter defaults cannot reference parameters before they are declared
//~| ERROR defaults for generic parameters are not allowed here
//~| ERROR the const parameter `N` is not constrained by the impl trait, self type, or predicates
//~| ERROR cannot define inherent `impl` for a type outside of the crate where the type is defined

fn foo() {}

pub struct S<const N: usize = { const { [1; foo] } }>;
//~^ ERROR mismatched types

fn main() {}
