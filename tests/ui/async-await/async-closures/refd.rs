//@ aux-build:block-on.rs
//@ edition:2021
//@ build-pass

#![feature(async_closure)]

extern crate block_on;

struct NoCopy;

fn main() {
    block_on::block_on(async {
        async fn call_once(x: impl async Fn()) { x().await }

        // check that `&{async-closure}` implements `async Fn`.
        call_once(&async || {}).await;

        // check that `&{closure}` implements `async Fn`.
        call_once(&|| async {}).await;

        // check that `&fndef` implements `async Fn`.
        async fn foo() {}
        call_once(&foo).await;
    });
}
