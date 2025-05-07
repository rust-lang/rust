//@ aux-build:block-on.rs
//@ edition:2021
//@ build-pass

extern crate block_on;

// Make sure that we don't call `coroutine_by_move_body_def_id` query
// on async closures that are `FnOnce`. See issue: #130167.

async fn empty() {}

pub async fn call_once<F: AsyncFnOnce()>(f: F) {
    f().await;
}

fn main() {
    block_on::block_on(call_once(async || empty().await));
}
