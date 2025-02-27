//@ check-pass
#![feature(sized_hierarchy)]

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
