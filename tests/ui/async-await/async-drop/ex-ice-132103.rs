//@ run-pass
//! This test used to ICE: rust-lang/rust#132103
//! Fixed when re-work async drop to shim drop glue coroutine scheme.
//@ compile-flags: -Zvalidate-mir -Zinline-mir=yes
//@ edition: 2018
#![feature(async_drop)]
#![allow(incomplete_features)]

use core::future::{async_drop_in_place, Future};
use core::mem::{self};
use core::pin::pin;
use core::task::{Context, Waker};

async fn test_async_drop<T>(x: T) {
    let mut x = mem::MaybeUninit::new(x);
    pin!(unsafe { async_drop_in_place(x.as_mut_ptr()) });
}

fn main() {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(&waker);

    let fut = pin!(async {
        test_async_drop(test_async_drop(0)).await;
    });
    let _ = fut.poll(&mut cx);
}
