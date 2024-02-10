// aux-build:block-on.rs
// edition:2021
// build-pass

#![feature(async_closure)]

use std::future::Future;

extern crate block_on;

struct NoCopy;

fn main() {
    block_on::block_on(async {
        async fn call_once<F: Future>(x: impl Fn(&'static str) -> F) -> F::Output {
            x("hello, world").await
        }
        call_once(async |x: &'static str| {
            println!("hello, {x}");
        }).await
    });
}
