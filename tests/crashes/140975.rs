//@ known-bug: #140975
//@ compile-flags: --crate-type lib -Zvalidate-mir
//@ edition: 2021
#![feature(async_drop)]
use std::{future::AsyncDrop, pin::Pin};

struct HasAsyncDrop ;
impl Drop for HasAsyncDrop {
    fn drop(&mut self) {}
}
impl AsyncDrop for HasAsyncDrop {
    async fn drop(self: Pin<&mut Self>) {}
}

struct Holder {
    inner: HasAsyncDrop,
}
async fn bar() {
    Holder {
        inner: HasAsyncDrop
   };
}
