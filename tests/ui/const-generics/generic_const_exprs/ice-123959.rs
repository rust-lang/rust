//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn foo<T>(_: [(); std::mem::offset_of!((T,), 0)]) {}

pub fn main() {}
