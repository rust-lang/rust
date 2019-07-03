// build-pass (FIXME(62277): could be check-pass?)
// edition:2018

#![feature(async_await, await_macro)]

use std::future::Future;

#[allow(unused)]
async fn foo<F: Future<Output = i32>>(x: &i32, future: F) -> i32 {
    let y = await!(future);
    *x + y
}

fn main() {}
