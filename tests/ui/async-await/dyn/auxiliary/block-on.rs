//@ edition: 2021

#![feature(async_closure, noop_waker)]

use std::future::Future;
use std::pin::pin;
use std::task::*;

pub fn block_on<T>(fut: impl Future<Output = T>) -> T {
    let mut fut = pin!(fut);
    // Poll loop, just to test the future...
    let ctx = &mut Context::from_waker(Waker::noop());

    loop {
        match unsafe { fut.as_mut().poll(ctx) } {
            Poll::Pending => {}
            Poll::Ready(t) => break t,
        }
    }
}
