//@ check-pass
//@ edition:2018

use std::future::Future;

async fn foo<F: Future<Output = i32>>(x: &i32, future: F) -> i32 {
    let y = future.await;
    *x + y
}

fn main() {}
