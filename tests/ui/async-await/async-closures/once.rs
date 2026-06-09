//@ aux-build:block-on.rs
//@ edition:2021
//@ build-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

use std::future::Future;

extern crate block_on;

// Check that async closures always implement `FnOnce`

fn main() {
    block_on::block_on(async {
        async fn call_once<F: Future>(x: impl FnOnce(&'static str) -> F) -> F::Output {
            x("hello, world").await
        }
        call_once(async |x: &'static str| {
            println!("hello, {x}");
        }).await
    });
}
