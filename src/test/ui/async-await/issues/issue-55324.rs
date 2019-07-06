// build-pass (FIXME(62277): could be check-pass?)
// edition:2018

#![feature(async_await)]

use std::future::Future;

#[allow(unused)]
async fn foo<F: Future<Output = i32>>(x: &i32, future: F) -> i32 {
    let y = future.await;
    *x + y
}

fn main() {}
