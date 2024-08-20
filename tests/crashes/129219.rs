//@ known-bug: rust-lang/rust#129219
//@ compile-flags: -Zmir-opt-level=5 -Zvalidate-mir --edition=2018

use core::marker::Unsize;

pub trait CastTo<T: ?Sized>: Unsize<T> {}

impl<T: ?Sized, U: ?Sized> CastTo<T> for U {}

impl<T: ?Sized> Cast for T {}
pub trait Cast {
    fn cast<T: ?Sized>(&self) -> &T
    where
        Self: CastTo<T>,
    {
        self
    }
}

pub trait Foo {}
impl Foo for [i32; 0] {}

fn main() {
    let x: &dyn Foo = &[];
    let x = x.cast::<[i32]>();
}
