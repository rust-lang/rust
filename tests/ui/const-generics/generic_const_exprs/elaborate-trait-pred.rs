//@ run-pass
// Test that we use the elaborated predicates from traits
// to satisfy const evaluatable predicates.
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
use std::mem::size_of;

trait Foo: Sized
where
    [(); size_of::<Self>()]: Sized,
{
}

impl Foo for u64 {}
impl Foo for u32 {}

fn foo<T: Foo>() -> [u8; size_of::<T>()] {
    [0; size_of::<T>()]
}

fn main() {
    assert_eq!(foo::<u32>(), [0; 4]);
    assert_eq!(foo::<u64>(), [0; 8]);
}
