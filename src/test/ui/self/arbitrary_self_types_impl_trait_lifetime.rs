// compile-fail

#![feature(arbitrary_self_types)]

use std::pin::Pin;

#[derive(Debug)]
struct Foo;
#[derive(Debug)]
struct Bar<'a>(&'a Foo);

impl std::ops::Deref for Bar<'_> {
    type Target = Foo;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Foo {
    fn f(self: Bar<'_>) -> impl std::fmt::Debug {
        self
        //~^ ERROR cannot infer an appropriate lifetime
    }
}

fn main() {
    { Bar(&Foo).f() };
}
