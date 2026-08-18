//! Regression test for <https://github.com/rust-lang/rust/issues/154805>.
//@ compile-flags: -Znext-solver=globally
#![feature(generic_const_items,min_generic_const_args)]
#![feature(macroless_generic_const_args, generic_const_args)]
const ADD1<const N: usize>: usize = N + 1;
fn a() -> [usize; ADD1::<b"">] {} //~ ERROR: type mismatch resolving `ADD1<*b""> == _` [E0271]
//~| ERROR: the type `[usize; ADD1::<*b"">]` is not well-formed
//~| ERROR: mismatched types [E0308]
//~| ERROR: type mismatch resolving `ADD1<*b""> == _` [E0271]


fn main() {}
