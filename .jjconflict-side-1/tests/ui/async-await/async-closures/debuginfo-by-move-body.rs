//@ aux-build:block-on.rs
//@ edition: 2021
//@ build-pass
//@ compile-flags: -Cdebuginfo=2

extern crate block_on;

async fn call_once(f: impl AsyncFnOnce()) {
    f().await;
}

pub fn main() {
    block_on::block_on(async {
        let async_closure = async move || {};
        call_once(async_closure).await;
    });
}
