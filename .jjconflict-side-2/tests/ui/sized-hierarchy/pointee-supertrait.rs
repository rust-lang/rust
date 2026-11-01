//@ check-pass
#![feature(sized_hierarchy)]

// This is a reduction of some code in `library/core/src/cmp.rs` that would ICE if a default
// `Pointee` bound is added - motivating the current status quo of `PointeeSized` being syntactic
// sugar for an absense of any bounds whatsoever.

use std::marker::PhantomData;

pub trait Bar<'a> {
    type Foo;
}

pub struct Foo<'a, T: Bar<'a>> {
    phantom: PhantomData<&'a T>,
}

impl<'a, 'b, T> PartialEq<Foo<'b, T>> for Foo<'a, T>
    where
        T: for<'c> Bar<'c>,
        <T as Bar<'a>>::Foo: PartialEq<<T as Bar<'b>>::Foo>,
{
    fn eq(&self, _: &Foo<'b, T>) -> bool {
        loop {}
    }
}

fn main() { }
