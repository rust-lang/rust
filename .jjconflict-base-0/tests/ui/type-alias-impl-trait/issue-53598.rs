#![feature(impl_trait_in_assoc_type)]

use std::fmt::Debug;

pub trait Foo {
    type Item: Debug;

    fn foo<T: Debug>(_: T) -> Self::Item;
}

#[derive(Debug)]
pub struct S<T>(std::marker::PhantomData<T>);

pub struct S2;

impl Foo for S2 {
    type Item = impl Debug;

    fn foo<T: Debug>(_: T) -> Self::Item {
        //~^ ERROR type parameter `T` is part of concrete type but not used in parameter list for the `impl Trait` type alias
        S::<T>(Default::default())
    }
}

fn main() {
    S2::foo(123);
}
