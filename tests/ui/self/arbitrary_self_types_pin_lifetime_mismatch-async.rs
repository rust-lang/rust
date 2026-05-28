//@ edition:2018

use std::pin::Pin;

struct Foo;

impl Foo {
    async fn a(self: Pin<&Foo>, f: &Foo) -> &Foo { f }
    //~^ ERROR lifetime may not live long enough

    async fn c(self: Pin<&Self>, f: &Foo, g: &Foo) -> (Pin<&Foo>, &Foo) { (self, f) }
    //~^ ERROR lifetime may not live long enough
}

type Alias<T> = Pin<T>;
impl Foo {
    async fn bar<'a>(self: Alias<&Self>, arg: &'a ()) -> &() { arg }
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
