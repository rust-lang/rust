// edition:2021
// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(async_closure, noop_waker, async_fn_traits)]

use std::future::Future;
use std::ops::{AsyncFnMut, AsyncFnOnce};
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

async fn call_mut(f: &mut impl AsyncFnMut(i32)) {
    f(0).await;
}

async fn call_once(f: impl AsyncFnOnce(i32)) {
    f(1).await;
}

// EMIT_MIR async_closure_shims.main-{closure#0}-{closure#0}.coroutine_closure_by_move.0.mir
// EMIT_MIR async_closure_shims.main-{closure#0}-{closure#0}.coroutine_closure_by_mut.0.mir
// EMIT_MIR async_closure_shims.main-{closure#0}-{closure#0}-{closure#0}.coroutine_by_mut.0.mir
// EMIT_MIR async_closure_shims.main-{closure#0}-{closure#0}-{closure#0}.coroutine_by_move.0.mir
fn main() {
    block_on(async {
        let b = 2i32;
        let mut async_closure = async move |a: i32| {
            let a = &a;
            let b = &b;
        };
        call_mut(&mut async_closure).await;
        call_once(async_closure).await;
    });
}
