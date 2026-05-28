//@ edition: 2018

#![feature(async_drop)]

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
