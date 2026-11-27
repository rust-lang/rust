//! This test checks that we don't follow up
//! with type mismatch errors of opaque types
//! with their hidden types if we failed the
//! defining scope check at the signature level.

#![feature(impl_trait_in_assoc_type)]

trait Foo {
    type Bar<T>;
    fn foo() -> Self::Bar<u32>;
    fn bar<T>() -> Self::Bar<T>;
}

impl Foo for () {
    type Bar<T> = impl Sized;
    fn foo() -> Self::Bar<u32> {}
    //~^ ERROR non-defining opaque type use
    //~| ERROR expected generic type parameter, found `u32`
    fn bar<T>() -> Self::Bar<T> {}
}

fn main() {}
