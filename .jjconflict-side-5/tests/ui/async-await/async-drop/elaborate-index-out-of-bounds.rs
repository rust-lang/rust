//@ edition: 2024
// Ex-ICE: #140974
#![crate_type = "lib"]
#![allow(incomplete_features)]
#![feature(async_drop)]
use core::future::AsyncDrop;

async fn fun(_: HasIncompleteAsyncDrop) {}

struct HasIncompleteAsyncDrop;
impl Drop for HasIncompleteAsyncDrop {
    fn drop(&mut self) {}
}
impl AsyncDrop for HasIncompleteAsyncDrop {
    //~^ ERROR: not all trait items implemented, missing: `drop` [E0046]
    // not implemented yet..
}
