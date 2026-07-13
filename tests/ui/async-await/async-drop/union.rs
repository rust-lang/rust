//@ edition: 2018

// `AsyncDrop` may not be implemented for unions: a union has no async drop glue,
// so dropping it in a synchronous context would silently skip the async cleanup.
// Synchronous `Drop` on a union remains allowed.
//
// This resolves the `FIXME(zetanumbers): Disallow types with AsyncDrop in unions`
// and implements the "union types rejecting non-trivially async destructible
// fields" item from the async drop tracking issue (rust-lang/rust#126482).

#![feature(async_drop)]
#![allow(incomplete_features)]

use std::future::AsyncDrop;
use std::pin::Pin;

union U {
    a: i32,
    b: u32,
}

// Allowed: a union may still implement the synchronous `Drop` trait.
impl Drop for U {
    fn drop(&mut self) {}
}

// Rejected: a union may not implement `AsyncDrop`.
impl AsyncDrop for U {
    //~^ ERROR `AsyncDrop` impl for `Union`
    async fn drop(self: Pin<&mut Self>) {}
}

fn main() {}
