//@ known-bug: rust-lang/rust#124894
//@ compile-flags: -Znext-solver=coherence

#![feature(generic_const_exprs)]

pub trait IsTrue<const mem: bool> {}
impl<T> IsZST for T where (): IsTrue<{ std::mem::size_of::<T>() == 0 }> {}

pub trait IsZST {}

impl IsZST for IsZST {}
