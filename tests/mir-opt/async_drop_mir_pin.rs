//@edition: 2024
//@ test-mir-pass: MentionedItems
// skip-filecheck
#![feature(async_drop)]
#![allow(incomplete_features)]
use std::future::AsyncDrop;
use std::pin::Pin;
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
// EMIT_MIR core.future-async_drop-async_drop_in_place-{closure#0}.[Foo;1].MentionedItems.after.mir
impl AsyncDrop for Foo {
    async fn drop(self: Pin<&mut Self>) {}
}
fn main() {}
async fn bar() {
    [Foo::new(3)];
}
