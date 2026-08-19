//! Regression test for <https://github.com/rust-lang/rust/issues/156780>.
//@ compile-flags: -Znext-solver=globally
#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
#![feature(generic_const_items)]
#![feature(macroless_generic_const_args)]

const ADD1<const N:usize>: usize = N + 1;
type const A<const N:usize>: usize = ADD1::<N>;
impl [(); A::<1f64>] {} //~ ERROR: type mismatch resolving `A<1f64> == _` [E0271]
//~| ERROR: cannot define inherent `impl` for primitive types [E0390]
      //~^^ ERROR: the type `[(); A::<1f64>]` is not well-formed
fn main() {}
