// edition:2018

use std::pin::Pin;

struct Foo;

impl Foo {
    async fn a(self: Pin<&Foo>, f: &Foo) -> &Foo { f }
    //~^ ERROR lifetime mismatch

    async fn c(self: Pin<&Self>, f: &Foo, g: &Foo) -> (Pin<&Foo>, &Foo) { (self, f) }
    //~^ ERROR lifetime mismatch
}

type Alias<T> = Pin<T>;
impl Foo {
    async fn bar<'a>(self: Alias<&Self>, arg: &'a ()) -> &() { arg } //~ ERROR E0623
}

fn main() {}
