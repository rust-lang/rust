// check-pass
#![allow(incomplete_features)]
#![feature(const_generics)]

pub trait Foo<const B: bool> {}
pub fn bar<T: Foo<{ true }>>() {}

fn main() {}
