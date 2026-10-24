//@ check-pass

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub struct Foo<const N: usize>;
pub trait Bar<T> {}

impl<T> Bar<T> for Foo<{ 1 }> {}
impl<T> Bar<T> for Foo<{ 2 }> {}

fn main() {}
