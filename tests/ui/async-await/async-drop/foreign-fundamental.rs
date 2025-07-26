//@ edition: 2018

#![allow(incomplete_features)]
#![feature(async_drop, async_drop_lib)]

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
