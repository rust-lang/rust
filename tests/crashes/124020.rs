//@ known-bug: #124020
//@ compile-flags: -Zpolymorphize=on --edition=2018 --crate-type=lib

#![feature(async_closure, noop_waker, async_trait_bounds)]

use std::future::Future;
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

async fn call_once(f: impl AsyncFnOnce(DropMe)) {
    f(DropMe("world")).await;
}

struct DropMe(&'static str);

pub fn future() {
    block_on(async {
        let async_closure = async move |a: DropMe| {};
        call_once(async_closure).await;
    });
}
