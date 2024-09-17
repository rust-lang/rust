//@ aux-build:block-on.rs
//@ edition: 2021
//@ build-pass
//@ compile-flags: -Cdebuginfo=2

#![feature(async_closure)]

extern crate block_on;

async fn call_once(f: impl async FnOnce()) {
    f().await;
}

pub fn main() {
    block_on::block_on(async {
        let async_closure = async move || {};
        call_once(async_closure).await;
    });
}
