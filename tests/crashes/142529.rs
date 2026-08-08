//@ known-bug: rust-lang/rust#142529
#![feature(min_generic_const_args)]
struct Foo<T, U = [u8; size_of::<T>]>(T, U);
fn main() {}
