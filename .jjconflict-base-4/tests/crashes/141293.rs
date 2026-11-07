//@ known-bug: #141293
#![feature(unsafe_binders)]
type X = unsafe<T> ();
fn main() {}
