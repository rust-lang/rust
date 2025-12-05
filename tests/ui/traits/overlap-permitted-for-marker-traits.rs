//@ run-pass
// Tests for RFC 1268: we allow overlapping impls of marker traits,
// that is, traits without items. In this case, a type `T` is
// `MyMarker` if it is either `Debug` or `Display`.

#![feature(marker_trait_attr)]
#![feature(negative_impls)]

use std::fmt::{Debug, Display};

#[marker]
trait MyMarker {}

impl<T: Debug> MyMarker for T {}
impl<T: Display> MyMarker for T {}

fn foo<T: MyMarker>(t: T) -> T {
    t
}

fn main() {
    // Debug && Display:
    assert_eq!(1, foo(1));
    assert_eq!(2.0, foo(2.0));

    // Debug && !Display:
    assert_eq!(vec![1], foo(vec![1]));
}
