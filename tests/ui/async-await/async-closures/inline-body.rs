//@ edition: 2021
//@ compile-flags: -Zinline-mir
//@ build-pass

// Ensure that we don't hit a Steal ICE because we forgot to ensure
// `mir_inliner_callees` for the synthetic by-move coroutine body since
// its def-id wasn't previously being considered.

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

async fn call_once<T>(f: impl AsyncFnOnce() -> T) -> T {
    f().await
}

fn main() {
    let c = async || {};
    block_on(call_once(c));
}
