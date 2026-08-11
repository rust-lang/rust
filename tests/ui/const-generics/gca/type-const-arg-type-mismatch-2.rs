//! Regression test for <https://github.com/rust-lang/rust/issues/154805>.
//@ compile-flags: -Znext-solver=globally
#![feature(min_generic_const_args)]
#![feature(macroless_generic_const_args)]
#![feature(generic_const_args)]
#![feature(generic_const_items)]

const ADD1<const N: usize>: usize = N + 1;
type const ONE: usize = ADD1::<b"">; //~ ERROR type mismatch resolving
//~| ERROR the constant `*b""` is not of type `usize`
//~| ERROR the constant `ADD1::<*b"">` is not of type `usize`
fn main() {}
