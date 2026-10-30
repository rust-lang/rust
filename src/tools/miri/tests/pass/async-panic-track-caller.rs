// This test is duplicated (with changes) at
// tests/ui/async-await/track-caller/panic-track-caller.rs

//@ edition:2024
//@ revisions: nofeat afn cls afn_cls
//@ compile-flags: -Zinline-mir-hint-threshold=1000
//
//
//
//
//
// Padding comment so that the line numbers are the same as panic-track-caller.rs
#![feature(stmt_expr_attributes, coroutines, coroutine_trait, gen_blocks)]
#![cfg_attr(any(afn, afn_cls), feature(async_fn_track_caller))]
#![cfg_attr(any(cls, afn_cls), feature(closure_track_caller))]
#![allow(unused)]

use std::future::Future;
use std::ops::Coroutine;
use std::panic::{self, Location};
use std::pin::pin;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
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

static LINE: AtomicU32 = AtomicU32::new(0);

async fn bar() {
    LINE.store(Location::caller().line(), Relaxed);
    #[cfg(panic = "unwind")]
    panic!()
}

async fn foo() {
    let future = bar();
    future.await;
}

#[cfg_attr(any(cls, nofeat), expect(ungated_async_fn_track_caller))]
#[track_caller]
async fn bar_track_caller() {
    LINE.store(Location::caller().line(), Relaxed);
    #[cfg(panic = "unwind")]
    panic!();
}

async fn foo_track_caller() {
    let future = bar_track_caller();
    future.await;
}

struct Foo;

impl Foo {
    #[cfg_attr(any(cls, nofeat), expect(ungated_async_fn_track_caller))]
    #[track_caller]
    async fn bar_assoc() {
        LINE.store(Location::caller().line(), Relaxed);
        #[cfg(panic = "unwind")]
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
        LINE.store(Location::caller().line(), Relaxed);
        #[cfg(panic = "unwind")]
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
        LINE.store(Location::caller().line(), Relaxed);
        #[cfg(panic = "unwind")]
        panic!();
    };
    future.await;
}

#[cfg_attr(any(cls, nofeat), expect(ungated_async_fn_track_caller))]
#[track_caller]
async fn bar_manual_poll() {
    LINE.store(Location::caller().line(), Relaxed);
    #[cfg(panic = "unwind")]
    panic!();
}

fn foo_manual_poll() {
    let future = bar_manual_poll();
    let future = std::pin::pin!(future);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let res = future.poll(&mut cx);
    assert_eq!(res, std::task::Poll::Ready(()));
}

trait Trait {
    async fn bar_trait_attr_nowhere();
    #[track_caller]
    async fn bar_trait_attr_in_trait();
    async fn bar_trait_attr_in_impl();
    #[track_caller]
    async fn bar_trait_attr_in_both();

    #[track_caller]
    fn bar_rpit_in_trait() -> impl Future<Output = ()>;
    #[track_caller]
    async fn bar_rpit_in_impl();
}
impl Trait for Foo {
    async fn bar_trait_attr_nowhere() {
        LINE.store(Location::caller().line(), Relaxed);
        #[cfg(panic = "unwind")]
        panic!();
    }
    async fn bar_trait_attr_in_trait() {
        LINE.store(Location::caller().line(), Relaxed);
        #[cfg(panic = "unwind")]
        panic!();
    }
    #[cfg_attr(any(cls, nofeat), expect(ungated_async_fn_track_caller))]
    #[track_caller]
    async fn bar_trait_attr_in_impl() {
        LINE.store(Location::caller().line(), Relaxed);
        #[cfg(panic = "unwind")]
        panic!();
    }
    #[cfg_attr(any(cls, nofeat), expect(ungated_async_fn_track_caller))]
    #[track_caller]
    async fn bar_trait_attr_in_both() {
        LINE.store(Location::caller().line(), Relaxed);
        #[cfg(panic = "unwind")]
        panic!();
    }

    async fn bar_rpit_in_trait() {
        LINE.store(Location::caller().line(), Relaxed);
        #[cfg(panic = "unwind")]
        panic!();
    }
    fn bar_rpit_in_impl() -> impl Future<Output = ()> {
        async {
            LINE.store(Location::caller().line(), Relaxed);
            #[cfg(panic = "unwind")]
            panic!();
        }
    }
}

async fn foo_trait_attr_nowhere() {
    let future = Foo::bar_trait_attr_nowhere();
    future.await;
}
async fn foo_trait_attr_in_trait() {
    let future = Foo::bar_trait_attr_in_trait();
    future.await;
}
async fn foo_trait_attr_in_impl() {
    let future = Foo::bar_trait_attr_in_impl();
    future.await;
}
async fn foo_trait_attr_in_both() {
    let future = Foo::bar_trait_attr_in_both();
    future.await;
}

async fn foo_rpit_in_trait() {
    let future = Foo::bar_rpit_in_trait();
    future.await;
}
async fn foo_rpit_in_impl() {
    let future = Foo::bar_rpit_in_impl();
    future.await;
}

#[track_caller]
gen fn bar_gen_fn() {
    LINE.store(Location::caller().line(), Relaxed);
    #[cfg(panic = "unwind")]
    panic!();
}

fn foo_gen_fn() {
    let mut iter = bar_gen_fn();
    let _ = iter.next();
}

// Since compilation is expected to fail for this fn when `closure_track_caller`
// is disabled, we test that separately in `async-closure-gate.rs`
#[cfg(any(cls, afn_cls))]
fn foo_gen_block() {
    let mut iter = #[track_caller]
    gen {
        LINE.store(Location::caller().line(), Relaxed);
        #[cfg(panic = "unwind")]
        panic!();
        yield ();
    };
    let _ = iter.next();
}

// Since compilation is expected to fail for this fn when `closure_track_caller`
// is disabled, we test that separately in `async-closure-gate.rs`
#[cfg(any(cls, afn_cls))]
fn foo_coroutine() {
    let coro = #[track_caller]
    #[coroutine]
    || {
        LINE.store(Location::caller().line(), Relaxed);
        #[cfg(panic = "unwind")]
        panic!();
        yield ();
    };
    let coro = std::pin::pin!(coro);
    let _ = coro.resume(());
}

fn assert_panicked_at(f: impl FnOnce() + panic::UnwindSafe, line: u32) {
    let loc = Arc::new(Mutex::new(None));

    let hook = panic::take_hook();
    {
        let loc = loc.clone();
        panic::set_hook(Box::new(move |info| {
            *loc.lock().unwrap() = info.location().map(|loc| loc.line())
        }));
    }
    let result = panic::catch_unwind(f);
    panic::set_hook(hook);
    #[cfg(panic = "unwind")]
    {
        assert!(result.is_err());
        assert_eq!(loc.lock().unwrap().unwrap(), line);
    }
    #[cfg(not(panic = "unwind"))]
    assert!(result.is_ok());
}

fn main() {
    assert_panicked_at(|| block_on(foo()), 61);
    assert_eq!(LINE.load(Relaxed), 59);

    #[cfg(any(afn, afn_cls))]
    assert_panicked_at(|| block_on(foo_track_caller()), 78);
    #[cfg(any(afn, afn_cls))]
    assert_eq!(LINE.load(Relaxed), 78);
    #[cfg(any(cls, nofeat))]
    assert_panicked_at(|| block_on(foo_track_caller()), 74);
    #[cfg(any(cls, nofeat))]
    assert_eq!(LINE.load(Relaxed), 72);

    #[cfg(any(afn, afn_cls))]
    assert_panicked_at(|| block_on(foo_assoc()), 95);
    #[cfg(any(afn, afn_cls))]
    assert_eq!(LINE.load(Relaxed), 95);
    #[cfg(any(cls, nofeat))]
    assert_panicked_at(|| block_on(foo_assoc()), 90);
    #[cfg(any(cls, nofeat))]
    assert_eq!(LINE.load(Relaxed), 88);

    // FIXME(closure_track_caller): Currently, #[track_caller] on an async closure
    // uses the location where the future is awaited or polled.
    // It should be changed to use the location where the closure is called,
    // so the behavior matches that of `async fn`.
    #[cfg(afn_cls)]
    assert_panicked_at(|| block_on(foo_closure()), 110);
    #[cfg(afn_cls)]
    assert_eq!(LINE.load(Relaxed), 110);
    // FIXME(closure_track_caller): if closure_track_caller is enabled, but
    // async_fn_track_caller is disabled, then #[track_caller] on async closures
    // silently do nothing. Either it should function, or we should emit a warning.
    // See #161961
    #[cfg(cls)]
    assert_panicked_at(|| block_on(foo_closure()), 107);
    #[cfg(cls)]
    assert_eq!(LINE.load(Relaxed), 105);

    #[cfg(any(cls, afn_cls))]
    assert_panicked_at(|| block_on(foo_block()), 123);
    #[cfg(any(cls, afn_cls))]
    assert_eq!(LINE.load(Relaxed), 123);

    #[cfg(any(afn, afn_cls))]
    assert_panicked_at(|| foo_manual_poll(), 135);
    #[cfg(any(afn, afn_cls))]
    assert_eq!(LINE.load(Relaxed), 135);
    #[cfg(any(cls, nofeat))]
    assert_panicked_at(|| foo_manual_poll(), 131);
    #[cfg(any(cls, nofeat))]
    assert_eq!(LINE.load(Relaxed), 129);

    assert_panicked_at(|| block_on(foo_trait_attr_nowhere()), 159);
    assert_eq!(LINE.load(Relaxed), 157);

    // FIXME(async_fn_track_caller): This case just currently doesn't work.
    #[cfg(any(afn, afn_cls))]
    assert_panicked_at(|| block_on(foo_trait_attr_in_trait()), 200);
    #[cfg(any(afn, afn_cls))]
    assert_eq!(LINE.load(Relaxed), 200);
    #[cfg(any(cls, nofeat))]
    assert_panicked_at(|| block_on(foo_trait_attr_in_trait()), 164);
    #[cfg(any(cls, nofeat))]
    assert_eq!(LINE.load(Relaxed), 162);

    #[cfg(any(afn, afn_cls))]
    assert_panicked_at(|| block_on(foo_trait_attr_in_impl()), 204);
    #[cfg(any(afn, afn_cls))]
    assert_eq!(LINE.load(Relaxed), 204);
    #[cfg(any(cls, nofeat))]
    assert_panicked_at(|| block_on(foo_trait_attr_in_impl()), 171);
    #[cfg(any(cls, nofeat))]
    assert_eq!(LINE.load(Relaxed), 169);

    #[cfg(any(afn, afn_cls))]
    assert_panicked_at(|| block_on(foo_trait_attr_in_both()), 208);
    #[cfg(any(afn, afn_cls))]
    assert_eq!(LINE.load(Relaxed), 208);
    #[cfg(any(cls, nofeat))]
    assert_panicked_at(|| block_on(foo_trait_attr_in_both()), 178);
    #[cfg(any(cls, nofeat))]
    assert_eq!(LINE.load(Relaxed), 176);

    #[cfg(any(afn, afn_cls))]
    assert_panicked_at(|| block_on(foo_rpit_in_trait()), 213);
    #[cfg(any(afn, afn_cls))]
    assert_eq!(LINE.load(Relaxed), 213);
    #[cfg(any(cls, nofeat))]
    assert_panicked_at(|| block_on(foo_rpit_in_trait()), 184);
    #[cfg(any(cls, nofeat))]
    assert_eq!(LINE.load(Relaxed), 182);

    assert_panicked_at(|| block_on(foo_rpit_in_impl()), 190);
    assert_eq!(LINE.load(Relaxed), 188);

    // FIXME(gen_blocks): Decide if this behavior is correct.
    #[cfg(any(afn, afn_cls))]
    assert_panicked_at(|| foo_gen_fn(), 229);
    #[cfg(any(afn, afn_cls))]
    assert_eq!(LINE.load(Relaxed), 229);
    #[cfg(any(cls, nofeat))]
    assert_panicked_at(|| foo_gen_fn(), 225);
    #[cfg(any(cls, nofeat))]
    assert_eq!(LINE.load(Relaxed), 223);

    #[cfg(any(cls, afn_cls))]
    assert_panicked_at(|| foo_gen_block(), 244);
    #[cfg(any(cls, afn_cls))]
    assert_eq!(LINE.load(Relaxed), 244);

    // FIXME(coroutines): This behavior is inconsistent with async blocks.
    #[cfg(any(cls, afn_cls))]
    assert_panicked_at(|| foo_coroutine(), 260);
    #[cfg(any(cls, afn_cls))]
    assert_eq!(LINE.load(Relaxed), 260);
}
