//@ known-bug: rust-lang/rust#142560
//@ edition:2021
//@ compile-flags: -Zlint-mir
#![feature(async_drop)]
use std::{future::AsyncDrop, pin::Pin};
struct Foo {
    my_resource_handle: usize,
}
impl Foo {
    fn new(my_resource_handle: usize) -> Self {
        let out = Foo { my_resource_handle };
        out
    }
}
impl Drop for Foo {
    fn drop(&mut self) {}
}
impl AsyncDrop for Foo {
    async fn drop(self: Pin<&mut Self>) {}
}
fn main() {}
async fn bar() {
    [Foo::new(3)];
}
