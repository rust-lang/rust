// Verifies that async closures can be called, including through dyn FnOnce
// trait objects.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ edition: 2021
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

#![feature(async_fn_traits)]

use std::future::Future;
use std::ops::AsyncFn;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

#[inline(never)]
fn identity<T>(x: T) -> T {
    x
}

fn poll<F: Future>(future: F) -> Poll<F::Output> {
    pin!(future).poll(&mut Context::from_waker(Waker::noop()))
}

fn main() {
    // The coroutine-closure is transformed into <dyn FnOnce() -> _ as FnOnce<()>>::call_once
    let f = identity(async || 1);
    assert_eq!(poll(f.async_call(())), Poll::Ready(1));
    assert_eq!(poll(f()), Poll::Ready(1));
    // The ConstructCoroutineInClosureShim and the VTableShim for
    // <{async closure} as FnOnce<()>>::call_once are transformed into
    // <dyn FnOnce() -> _ as FnOnce<()>>::call_once.
    let g: Box<dyn FnOnce() -> _> = Box::new(f) as _;
    // The virtual method call is transformed into <dyn FnOnce() -> _ as FnOnce<()>>::call_once
    assert_eq!(poll(g()), Poll::Ready(1));
}
