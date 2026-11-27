// Regression test for #121094.
//@ build-pass
//@ compile-flags: -O --crate-type=lib
//@ edition: 2021
use std::{future::Future, pin::Pin};

pub async fn foo(count: u32) {
    if count == 0 {
        return
    } else {
        let fut: Pin<Box<dyn Future<Output = ()>>> = Box::pin(foo(count - 1));
        fut.await;
    }
}
