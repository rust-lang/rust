//@ compile-flags: -Znext-solver=globally
// Regression test for #156780.
// A float literal passed as a const arg where `usize` is expected
// should produce a type error, not an ICE.

#![feature(min_generic_const_args, generic_const_args, generic_const_items)]

type const ADD1<const N: usize>: usize = const { N + 1 };

impl [(); ADD1::<1f64>] {}
//~^ ERROR the literal is not of type `usize`
//~| ERROR cannot define inherent `impl` for primitive types

fn main() {}
