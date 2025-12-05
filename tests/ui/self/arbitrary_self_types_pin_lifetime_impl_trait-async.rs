//@ edition:2018

use std::pin::Pin;

struct Foo;

impl Foo {
    async fn f(self: Pin<&Self>) -> impl Clone { self }
    //~^ ERROR: captures lifetime that does not appear in bounds
}

fn main() {
    { Pin::new(&Foo).f() };
}
