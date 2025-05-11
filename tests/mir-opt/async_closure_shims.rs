//@ edition:2021
// skip-filecheck

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

async fn call(f: &impl AsyncFn(i32)) {
    f(0).await;
}

async fn call_mut(f: &mut impl AsyncFnMut(i32)) {
    f(0).await;
}

async fn call_once(f: impl AsyncFnOnce(i32)) {
    f(1).await;
}

async fn call_normal<F: Future<Output = ()>>(f: &impl Fn(i32) -> F) {
    f(1).await;
}

async fn call_normal_mut<F: Future<Output = ()>>(f: &mut impl FnMut(i32) -> F) {
    f(1).await;
}

// EMIT_MIR async_closure_shims.main-{closure#0}-{closure#0}.coroutine_closure_by_move.0.mir
// EMIT_MIR async_closure_shims.main-{closure#0}-{closure#0}-{closure#0}.built.after.mir
// EMIT_MIR async_closure_shims.main-{closure#0}-{closure#0}-{synthetic#0}.built.after.mir
// EMIT_MIR async_closure_shims.main-{closure#0}-{closure#1}.coroutine_closure_by_ref.0.mir
// EMIT_MIR async_closure_shims.main-{closure#0}-{closure#1}.coroutine_closure_by_move.0.mir
// EMIT_MIR async_closure_shims.main-{closure#0}-{closure#1}-{closure#0}.built.after.mir
// EMIT_MIR async_closure_shims.main-{closure#0}-{closure#1}-{synthetic#0}.built.after.mir
pub fn main() {
    block_on(async {
        let b = 2i32;
        let mut async_closure = async move |a: i32| {
            let a = &a;
            let b = &b;
        };
        call(&async_closure).await;
        call_mut(&mut async_closure).await;
        call_once(async_closure).await;

        let b = 2i32;
        let mut async_closure = async |a: i32| {
            let a = &a;
            let b = &b;
        };
        call_normal(&async_closure).await;
        call_normal_mut(&mut async_closure).await;
        call_once(async_closure).await;
    });
}
