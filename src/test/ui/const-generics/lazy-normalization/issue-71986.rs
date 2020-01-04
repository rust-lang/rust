// check-pass
#![allow(incomplete_features)]
#![feature(const_generics, lazy_normalization_consts)]

pub trait Foo<const B: bool> {}
pub fn bar<T: Foo<{ true }>>() {}

fn main() {}
