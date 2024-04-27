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

async fn call_normal<F: Future<Output = ()>>(f: &impl Fn(i32) -> F) {
    f(0).await;
}

async fn call_normal_once<F: Future<Output = ()>>(f: impl FnOnce(i32) -> F) {
    f(1).await;
}

pub fn main() {
    block_on(async {
        let b = 2i32;
        let mut async_closure = async move |a: i32| {
            println!("{a} {b}");
        };
        call_mut(&mut async_closure).await;
        call_once(async_closure).await;

        // No-capture closures implement `Fn`.
        let async_closure = async move |a: i32| {
            println!("{a}");
        };
        call_normal(&async_closure).await;
        call_normal_once(async_closure).await;
    });
}
