//@ aux-build:block-on.rs
//@ edition:2021
//@ build-pass

// check that `&{async-closure}` implements `AsyncFn`.

#![feature(async_closure)]

extern crate block_on;

struct NoCopy;

fn main() {
    block_on::block_on(async {
        async fn call_once(x: impl async Fn()) { x().await }
        call_once(&async || {}).await
    });
}
