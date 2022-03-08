#![feature(arbitrary_self_types)]
#![feature(type_alias_impl_trait)]

use std::ops::Deref;

trait Foo {
    type Bar: Foo;

    fn foo(self: impl Deref<Target = Self>) -> Self::Bar;
}

impl<C> Foo for C {
    type Bar = impl Foo;

    fn foo(self: impl Deref<Target = Self>) -> Self::Bar {
        //~^ Error type parameter `impl Deref<Target = Self>` is part of concrete type but not used in parameter list for the `impl Trait` type alias
        self
    }
}

fn main() {}
