//@ known-bug: #134175
//@compile-flags: -Zvalidate-mir -Zinline-mir=yes
use std::vec::IntoIter;

pub(crate) trait Foo: Iterator<Item = <Self as Foo>::Key> {
    type Key;
}

impl Foo for IntoIter<i16> {}

fn sum_foo<F: Foo<Key = i32>>(f: F) -> i32 {
    f.fold(0, |a, b| a + b)
}

fn main() {
    let x = sum_foo(vec![11, 10, 1].into_iter());
}
