// Regression test for <https://github.com/rust-lang/rust/issues/151135>.

//@ edition: 2021

//@ aux-build: executor.rs
extern crate executor;

use std::sync::atomic::{AtomicUsize, Ordering};

static STEPS: AtomicUsize = AtomicUsize::new(0);

async fn call_once(f: impl AsyncFnOnce()) {
    f().await;
}

pub fn main() {
    let async_closure = async || {
        STEPS.fetch_add(1, Ordering::SeqCst);
        STEPS.fetch_add(1, Ordering::SeqCst);
    };
    executor::block_on(call_once(async_closure));
    assert_eq!(STEPS.load(Ordering::SeqCst), 2);
}
