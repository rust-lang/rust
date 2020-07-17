// edition:2018

use std::pin::Pin;

struct Foo;

impl Foo {
    async fn f(self: Pin<&Self>) -> impl Clone { self }
    //~^ ERROR E0759
}

fn main() {
    { Pin::new(&Foo).f() };
}
