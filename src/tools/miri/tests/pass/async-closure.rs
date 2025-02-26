#![feature(async_fn_traits)]
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

async fn call(f: &mut impl AsyncFn(i32)) {
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

pub fn main() {
    block_on(async {
        let b = 2i32;
        let mut async_closure = async move |a: i32| {
            println!("{a} {b}");
        };
        call(&mut async_closure).await;
        call_mut(&mut async_closure).await;
        call_once(async_closure).await;

        let b = 2i32;
        let mut async_closure = async |a: i32| {
            println!("{a} {b}");
        };
        call_normal(&async_closure).await;
        call_normal_mut(&mut async_closure).await;
        call_once(async_closure).await;
    });
}
