//@ edition:2021
// skip-filecheck
// EMIT_MIR async_self_referential_fields.my_async_fn-{closure#0}.StateTransform.after.mir

#![allow(unused)]

use std::future::Future;
use std::ops::{AsyncFn, AsyncFnMut, AsyncFnOnce};
use std::pin::pin;
use std::task::*;

pub fn block_on<T>(fut: impl Future<Output = T>) -> T {
    let mut fut = pin!(fut);
    let ctx = &mut Context::from_waker(Waker::noop());

    loop {
        match fut.as_mut().poll(ctx) {
            Poll::Pending => {}
            Poll::Ready(t) => break t,
        }
    }
}

async fn inner_async_fn() {}

async fn my_async_fn() -> i32 {
    let x = Box::new(5);
    let y = &x;
    inner_async_fn().await;
    std::hint::black_box(y);
    *x + 1
}

fn main() {
    block_on(async {
        my_async_fn().await;
    });
}
