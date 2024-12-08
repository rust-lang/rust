//@ known-bug: #123959
#![feature(generic_const_exprs)]
fn foo<T>(_: [(); std::mem::offset_of!((T,), 0)]) {}

pub fn main() {}
