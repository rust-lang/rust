//@ edition: 2018

#![feature(async_drop)]
//~^ WARN the feature `async_drop` is incomplete

use std::future::AsyncDrop;
use std::pin::Pin;

struct Foo;

impl AsyncDrop for &Foo {
    //~^ ERROR the `AsyncDrop` trait may only be implemented for
    async fn drop(self: Pin<&mut Self>) {}
}

impl AsyncDrop for Pin<Foo> {
    //~^ ERROR the `AsyncDrop` trait may only be implemented for
    async fn drop(self: Pin<&mut Self>) {}
}

fn main() {}
