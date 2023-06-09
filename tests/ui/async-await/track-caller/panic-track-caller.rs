// run-pass
// edition:2021
// revisions: feat nofeat
// needs-unwind
#![feature(async_closure, stmt_expr_attributes)]
#![cfg_attr(feat, feature(closure_track_caller))]

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

#[track_caller] //[nofeat]~ WARN `#[track_caller]` on async functions is a no-op
async fn bar_track_caller() {
    panic!()
}

async fn foo_track_caller() {
    bar_track_caller().await
}

struct Foo;

impl Foo {
    #[track_caller] //[nofeat]~ WARN `#[track_caller]` on async functions is a no-op
    async fn bar_assoc() {
        panic!();
    }
}

async fn foo_assoc() {
    Foo::bar_assoc().await
}

// Since compilation is expected to fail for this fn when using
// `nofeat`, we test that separately in `async-closure-gate.rs`
#[cfg(feat)]
async fn foo_closure() {
    let c = #[track_caller] async || {
        panic!();
    };
    c().await
}

// Since compilation is expected to fail for this fn when using
// `nofeat`, we test that separately in `async-block.rs`
#[cfg(feat)]
async fn foo_block() {
    let a = #[track_caller] async {
        panic!();
    };
    a.await
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
    assert_eq!(panicked_at(|| block_on(foo())), 43);

    #[cfg(feat)]
    assert_eq!(panicked_at(|| block_on(foo_track_caller())), 56);
    #[cfg(nofeat)]
    assert_eq!(panicked_at(|| block_on(foo_track_caller())), 52);

    #[cfg(feat)]
    assert_eq!(panicked_at(|| block_on(foo_assoc())), 69);
    #[cfg(nofeat)]
    assert_eq!(panicked_at(|| block_on(foo_assoc())), 64);

    #[cfg(feat)]
    assert_eq!(panicked_at(|| block_on(foo_closure())), 79);

    #[cfg(feat)]
    assert_eq!(panicked_at(|| block_on(foo_block())), 89);
}
