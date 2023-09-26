// edition: 2021

#![feature(async_fn_in_trait)]
#![deny(async_fn_in_trait)]

trait Foo {
    async fn not_send();
    //~^ ERROR
}

fn main() {}
