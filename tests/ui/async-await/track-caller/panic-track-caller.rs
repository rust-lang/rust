//@ run-pass
//@ edition:2021
//@ revisions: afn cls afn_cls nofeat
//@ needs-unwind
// gate-test-async_fn_track_caller
#![feature(stmt_expr_attributes)]
#![cfg_attr(any(afn, afn_cls), feature(async_fn_track_caller))]
#![cfg_attr(any(cls, afn_cls), feature(closure_track_caller))]
#![allow(unused)]

use std::future::Future;
use std::panic;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake};
use std::thread::{self, Thread};

/// A waker that wakes up the current thread when called.
struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// Run a future to completion on the current thread.
fn block_on<T>(fut: impl Future<Output = T>) -> T {
    // Pin the future so it can be polled.
    let mut fut = Box::pin(fut);

    // Create a new context to be passed to the future.
    let t = thread::current();
    let waker = Arc::new(ThreadWaker(t)).into();
    let mut cx = Context::from_waker(&waker);

    // Run the future to completion.
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(res) => return res,
            Poll::Pending => thread::park(),
        }
    }
}

async fn bar() {
    panic!()
}

async fn foo() {
    let future = bar();
    future.await;
}

#[track_caller]
//[cls,nofeat]~^ WARN `#[track_caller]` on async functions is a no-op
async fn bar_track_caller() {
    panic!()
}

async fn foo_track_caller() {
    let future = bar_track_caller();
    future.await;
}

struct Foo;

impl Foo {
    #[track_caller]
    //[cls,nofeat]~^ WARN `#[track_caller]` on async functions is a no-op
    async fn bar_assoc() {
        panic!();
    }
}

async fn foo_assoc() {
    let future = Foo::bar_assoc();
    future.await;
}

// Since compilation is expected to fail for this fn when `closure_track_caller`
// is disabled, we test that separately in `async-closure-gate.rs`
#[cfg(any(cls, afn_cls))]
async fn foo_closure() {
    let closure = #[track_caller]
    async || {
        panic!();
    };
    let future = closure();
    future.await;
}

// Since compilation is expected to fail for this fn when `closure_track_caller`
// is disabled, we test that separately in `async-closure-gate.rs`
#[cfg(any(cls, afn_cls))]
async fn foo_block() {
    let future = #[track_caller]
    async {
        panic!();
    };
    future.await;
}

fn panicked_at(f: impl FnOnce() + panic::UnwindSafe) -> u32 {
    let loc = Arc::new(Mutex::new(None));

    let hook = panic::take_hook();
    {
        let loc = loc.clone();
        panic::set_hook(Box::new(move |info| {
            *loc.lock().unwrap() = info.location().map(|loc| loc.line())
        }));
    }
    panic::catch_unwind(f).unwrap_err();
    panic::set_hook(hook);
    let x = loc.lock().unwrap().unwrap();
    x
}

// FIXME(async_fn_track_caller): Currently, #[track_caller] on an async function
// uses the location where the future is awaited.
// The correct behavior as per T-lang is to use the location where the function is called.
fn main() {
    assert_eq!(panicked_at(|| block_on(foo())), 46);

    #[cfg(any(afn, afn_cls))]
    assert_eq!(panicked_at(|| block_on(foo_track_caller())), 62);
    #[cfg(any(cls, nofeat))]
    assert_eq!(panicked_at(|| block_on(foo_track_caller())), 57);

    #[cfg(any(afn, afn_cls))]
    assert_eq!(panicked_at(|| block_on(foo_assoc())), 77);
    #[cfg(any(cls, nofeat))]
    assert_eq!(panicked_at(|| block_on(foo_assoc())), 71);

    // FIXME(closure_track_caller): if closure_track_caller is enabled, but
    // async_fn_track_caller is disabled, then #[track_caller] on async closures
    // silently do nothing. Either it should function, or we should emit a warning.
    // See #161961
    #[cfg(cls)]
    assert_eq!(panicked_at(|| block_on(foo_closure())), 86);
    #[cfg(afn_cls)]
    assert_eq!(panicked_at(|| block_on(foo_closure())), 89);

    #[cfg(any(cls, afn_cls))]
    assert_eq!(panicked_at(|| block_on(foo_block())), 100);
}
