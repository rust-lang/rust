// build-pass

#![feature(generic_const_exprs)]
#![allow(unused_braces, incomplete_features)]

pub trait Foo<const N: usize> {}
pub trait Bar: Foo<{ 1 }> { }

fn main() {}
