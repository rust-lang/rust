// edition:2018
#![feature(async_await)]

// Regression test for issue #62382

use std::future::Future;

fn get_future() -> impl Future<Output = ()> {
    panic!()
}

async fn foo() {
    let a; //~ ERROR type inside `async` object must be known in this context
    get_future().await;
}

fn main() {}
