//! This test checks that we don't follow up
//! with type mismatch errors of opaque types
//! with their hidden types if we failed the
//! defining scope check at the signature level.

#![feature(impl_trait_in_assoc_type)]

trait Foo {
    type Bar<T>;
    type Baz;
    fn foo() -> (Self::Bar<u32>, Self::Baz);
}

impl Foo for () {
    type Bar<T> = impl Sized;
    type Baz = impl Sized;
    fn foo() -> (Self::Bar<u32>, Self::Baz) {
        //~^ ERROR non-defining opaque type use
        ((), ())
    }
}

fn main() {}
