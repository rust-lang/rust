// edition:2018

#![feature(async_await)]

use std::pin::Pin;

struct Foo;

impl Foo {
    async fn a(self: Pin<&Foo>, f: &Foo) -> &Foo { f }
    //~^ ERROR missing lifetime specifier
    //~| ERROR cannot infer an appropriate lifetime
    // FIXME: should be E0623?

    async fn c(self: Pin<&Self>, f: &Foo, g: &Foo) -> (Pin<&Foo>, &Foo) { (self, f) }
    //~^ ERROR missing lifetime specifier
    //~| ERROR cannot infer an appropriate lifetime
    //~| ERROR missing lifetime specifier
    //~| ERROR cannot infer an appropriate lifetime
    // FIXME: should be E0623?
}

type Alias<T> = Pin<T>;
impl Foo {
    async fn bar<'a>(self: Alias<&Self>, arg: &'a ()) -> &() { arg } //~ ERROR E0623
}

fn main() {}
