use std::{
    ops::Deref,
};

struct Ptr<T: ?Sized>(Box<T>);

impl<T: ?Sized> Deref for Ptr<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &*self.0
    }
}

trait Foo {
    fn foo(self: Ptr<Self>); //~ ERROR `Ptr<Self>` cannot be used as the type of `self` without
}

struct Bar;

impl Foo for Bar {
    fn foo(self: Ptr<Self>) {} //~ ERROR `Ptr<Bar>` cannot be used as the type of `self` without
}

impl Bar {
    fn bar(self: Box<Ptr<Self>>) {} //~ ERROR `Box<Ptr<Bar>>` cannot be used as the
}

fn main() {}
