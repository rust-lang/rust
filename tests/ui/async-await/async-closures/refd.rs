//@ aux-build:block-on.rs
//@ edition:2021
//@ build-pass

extern crate block_on;

struct NoCopy;

fn main() {
    block_on::block_on(async {
        async fn call_once(x: impl AsyncFn()) { x().await }

        // check that `&{async-closure}` implements `AsyncFn`.
        call_once(&async || {}).await;

        // check that `&{closure}` implements `AsyncFn`.
        call_once(&|| async {}).await;

        // check that `&fndef` implements `AsyncFn`.
        async fn foo() {}
        call_once(&foo).await;
    });
}
