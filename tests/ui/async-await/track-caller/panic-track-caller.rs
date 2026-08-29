//@ run-pass
//@ edition:2021
//@ revisions: cls nofeat
//@ needs-unwind
#![feature(stmt_expr_attributes)]
#![cfg_attr(cls, feature(closure_track_caller))]
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
    bar().await
}

#[track_caller]
async fn bar_track_caller() {
    panic!()
}

async fn foo_track_caller() {
    bar_track_caller().await
}

struct Foo;

impl Foo {
    #[track_caller]
    async fn bar_assoc() {
        panic!();
    }
}

async fn foo_assoc() {
    Foo::bar_assoc().await
}

// Since compilation is expected to fail for this fn when using
// `nofeat`, we test that separately in `async-closure-gate.rs`
#[cfg(cls)]
async fn foo_closure() {
    let c = #[track_caller]
    async || {
        panic!();
    };
    c().await
}

// Since compilation is expected to fail for this fn when using
// `nofeat`, we test that separately in `async-block.rs`
#[cfg(cls)]
async fn foo_block() {
    let a = #[track_caller]
    async {
        panic!();
    };
    a.await
}

#[track_caller]
async fn bar_manual_poll() {
    panic!();
}

fn foo_manual_poll() {
    let future = bar_manual_poll();
    let future = std::pin::pin!(future);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let res = future.poll(&mut cx);
    assert_eq!(res, std::task::Poll::Ready(()));
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

fn main() {
    assert_eq!(panicked_at(|| block_on(foo())), 44);

    assert_eq!(panicked_at(|| block_on(foo_track_caller())), 57);

    assert_eq!(panicked_at(|| block_on(foo_assoc())), 70);

    #[cfg(cls)]
    assert_eq!(panicked_at(|| block_on(foo_closure())), 81);

    #[cfg(cls)]
    assert_eq!(panicked_at(|| block_on(foo_block())), 92);

    // This should be 101 (call site), not 104 (poll site)
    assert_eq!(panicked_at(|| foo_manual_poll()), 104);
}
