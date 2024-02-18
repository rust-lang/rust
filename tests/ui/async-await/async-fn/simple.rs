//@ edition: 2021
//@ build-pass

#![feature(async_fn_traits)]

use std::ops::AsyncFn;

async fn foo() {}

async fn call_asyncly(f: impl AsyncFn(i32) -> i32) -> i32 {
    f(1).await
}

fn main() {
    let fut = call_asyncly(|x| async move { x + 1 });
}
