//@ edition: 2021

use std::future::AsyncDrop; //~ ERROR  use of unstable library feature `async_drop`
use std::pin::Pin;

struct Foo {}

impl Drop for Foo {
    fn drop(&mut self) {}
}

impl AsyncDrop for Foo { //~ ERROR  use of unstable library feature `async_drop`
    async fn drop(self: Pin<&mut Self>) {} //~ ERROR  use of unstable library feature `async_drop`
}

fn main() {}
