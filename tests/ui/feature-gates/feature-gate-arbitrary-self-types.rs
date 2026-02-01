use std::ops::Deref;

struct Ptr<T: ?Sized>(Box<T>);

impl<T: ?Sized> Deref for Ptr<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &*self.0
    }
}

trait Foo {
    fn foo(self: Ptr<Self>); //~ ERROR invalid `self` parameter type: `Ptr<Self>`
}

struct Bar;

impl Foo for Bar {
    fn foo(self: Ptr<Self>) {} //~ ERROR invalid `self` parameter type: `Ptr<Bar>`
}

impl Bar {
    fn bar(self: Box<Ptr<Self>>) {} //~ ERROR invalid `self` parameter type: `Box<Ptr<Bar>>`
}

fn main() {}
