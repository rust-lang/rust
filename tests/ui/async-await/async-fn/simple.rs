//@ aux-build:block-on.rs
//@ edition: 2021
//@ build-pass

#![feature(async_closure)]

extern crate block_on;

use std::ops::AsyncFn;

async fn foo() {}

async fn call_asyncly(f: impl AsyncFn(i32) -> i32) -> i32 {
    f(1).await
}

fn main() {
    block_on::block_on(async {
        call_asyncly(|x| async move { x + 1 }).await;
    });
}
