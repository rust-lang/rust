// edition: 2021
// check-pass

#![feature(async_fn_traits)]

use std::ops::AsyncFn;

async fn foo() {}

async fn call_asyncly(f: impl AsyncFn(i32) -> i32) -> i32 {
    f.async_call((1i32,)).await
}

fn main() {
    let fut = call_asyncly(|x| async move { x + 1 });
}
