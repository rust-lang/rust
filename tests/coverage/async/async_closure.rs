//@ edition: 2021

//@ aux-build: executor.rs
extern crate executor;

async fn call_once(f: impl AsyncFnOnce()) {
    f().await;
}

pub fn main() {
    let async_closure = async || {};
    executor::block_on(async_closure());
    executor::block_on(call_once(async_closure));
}
