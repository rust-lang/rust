//! This test shows that the appearance of an opaque type
//! in the substs of a struct are enough to make it count
//! for making the function a defining use. It doesn't matter
//! if the opaque type is actually used in the field.

#![feature(impl_trait_in_assoc_type)]
// check-pass

use std::marker::PhantomData;

struct Bar;

trait Trait: Sized {
    type Assoc;
    fn foo() -> Foo<Self::Assoc>;
}

impl Trait for Bar {
    type Assoc = impl std::fmt::Debug;
    fn foo() -> Foo<Self::Assoc> {
        let foo: Foo<()> = Foo { field: PhantomData };
        foo
    }
}

struct Foo<T> {
    field: PhantomData<T>,
}

fn main() {}
