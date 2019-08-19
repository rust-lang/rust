// Test that we don't consider temporaries for statement expressions as live
// across yields

// check-pass
// edition:2018

#![feature(async_await, generators, generator_trait)]

use std::ops::Generator;

async fn drop_and_await() {
    async {};
    async {}.await;
}

fn drop_and_yield() {
    let x = || {
        String::new();
        yield;
    };
    Box::pin(x).as_mut().resume();
    let y = static || {
        String::new();
        yield;
    };
    Box::pin(y).as_mut().resume();
}

fn main() {
    drop_and_await();
    drop_and_yield();
}
