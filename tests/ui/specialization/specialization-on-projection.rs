//@ run-pass
#![allow(dead_code)]

#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

// Ensure that specialization works for impls defined directly on a projection

trait Foo<T> {}

trait Assoc {
    type Item;
}

impl<T: Assoc> Foo<T::Item> for T {}

struct Struct;

impl Assoc for Struct {
    type Item = u8;
}

impl Foo<u8> for Struct {}

fn main() {}
