//! This test checks that we don't allow registering hidden types for
//! opaque types from other impls.

#![feature(impl_trait_in_assoc_type)]

trait Foo<T> {
    type Bar;
    fn foo(&self) -> <Self as Foo<()>>::Bar
    where
        Self: Foo<()>;
}

impl Foo<()> for () {
    type Bar = impl std::fmt::Debug;
    fn foo(&self) -> Self::Bar {}
}

impl Foo<i32> for () {
    type Bar = u32;
    fn foo(&self) -> <Self as Foo<()>>::Bar {}
    //~^ ERROR: mismatched types
}

fn main() {}
