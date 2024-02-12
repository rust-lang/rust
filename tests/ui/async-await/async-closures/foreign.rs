// aux-build:block-on.rs
// aux-build:foreign.rs
// edition:2021
// build-pass

#![feature(async_closure)]

use std::future::Future;

extern crate block_on;
extern crate foreign;

struct NoCopy;

fn main() {
    block_on::block_on(async {
        foreign::closure()().await;
    });
}
