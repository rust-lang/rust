// edition:2018

#![feature(async_await)]

use std::pin::Pin;

struct Foo;

impl Foo {
    async fn f(self: Pin<&Self>) -> impl Clone { self }
    //~^ ERROR cannot infer an appropriate lifetime
}

fn main() {
    { Pin::new(&Foo).f() };
}
