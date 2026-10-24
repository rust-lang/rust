// Ex-ICE: #140975
//@ compile-flags: -Zvalidate-mir
//@ build-pass
//@ edition:2021
#![crate_type = "lib"]
#![feature(async_drop)]
#![allow(incomplete_features)]

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
