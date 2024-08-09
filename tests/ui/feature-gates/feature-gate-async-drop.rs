use std::future::AsyncDrop;
use std::pin::Pin;

struct Foo {}

impl Drop for Foo {
    fn drop(&mut self) {}
}

impl AsyncDrop for Foo {
    async fn drop(self: Pin<&mut Self>) {}
}

fn main() {}
