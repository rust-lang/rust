//@ known-bug: rust-lang/rust#141293
#![feature(unsafe_binders)]
type X = unsafe<T> ();

type Y = unsafe<const N: i32> ();

pub fn main() {}
