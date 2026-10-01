// This test is duplicated (with changes) at
// src/tools/miri/tests/pass/async-panic-track-caller.rs

//@ run-pass
//@ edition:2024
//@ revisions: nofeat afn cls afn_cls nofeat_opt afn_opt cls_opt afn_cls_opt
//@[nofeat_opt] compile-flags: -O
//@[afn_opt] compile-flags: -O
//@[cls_opt] compile-flags: -O
//@[afn_cls_opt] compile-flags: -O
//@ needs-unwind
// gate-test-async_fn_track_caller
#![feature(stmt_expr_attributes, coroutines, coroutine_trait, gen_blocks)]
#![cfg_attr(any(afn, afn_cls, afn_opt, afn_cls_opt), feature(async_fn_track_caller))]
#![cfg_attr(any(cls, afn_cls, cls_opt, afn_cls_opt), feature(closure_track_caller))]
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
    panic!()
}

async fn foo() {
    let future = bar();
    future.await;
}

#[track_caller]
//[nofeat,cls,nofeat_opt,cls_opt]~^ WARN `#[track_caller]` on async functions is a no-op
async fn bar_track_caller() {
    LINE.store(Location::caller().line(), Relaxed);
    panic!()
}

async fn foo_track_caller() {
    let future = bar_track_caller();
    future.await;
}

struct Foo;

impl Foo {
    #[track_caller]
    //[nofeat,cls,nofeat_opt,cls_opt]~^ WARN `#[track_caller]` on async functions is a no-op
    async fn bar_assoc() {
        LINE.store(Location::caller().line(), Relaxed);
        panic!();
    }
}

async fn foo_assoc() {
    let future = Foo::bar_assoc();
    future.await;
}

// Since compilation is expected to fail for this fn when `closure_track_caller`
// is disabled, we test that separately in `async-closure-gate.rs`
#[cfg(any(cls, afn_cls, cls_opt, afn_cls_opt))]
async fn foo_closure() {
    let closure = #[track_caller]
    async || {
        LINE.store(Location::caller().line(), Relaxed);
        panic!();
    };
    let future = closure();
    future.await;
}

// Since compilation is expected to fail for this fn when `closure_track_caller`
// is disabled, we test that separately in `async-closure-gate.rs`
#[cfg(any(cls, afn_cls, cls_opt, afn_cls_opt))]
async fn foo_block() {
    let future = #[track_caller]
    async {
        LINE.store(Location::caller().line(), Relaxed);
        panic!();
    };
    future.await;
}

#[track_caller]
//[nofeat,cls,nofeat_opt,cls_opt]~^ WARN `#[track_caller]` on async functions is a no-op
async fn bar_manual_poll() {
    LINE.store(Location::caller().line(), Relaxed);
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
        panic!();
    }
    async fn bar_trait_attr_in_trait() {
        LINE.store(Location::caller().line(), Relaxed);
        panic!();
    }
    #[track_caller]
    //[nofeat,cls,nofeat_opt,cls_opt]~^ WARN `#[track_caller]` on async functions is a no-op
    async fn bar_trait_attr_in_impl() {
        LINE.store(Location::caller().line(), Relaxed);
        panic!();
    }
    #[track_caller]
    //[nofeat,cls,nofeat_opt,cls_opt]~^ WARN `#[track_caller]` on async functions is a no-op
    async fn bar_trait_attr_in_both() {
        LINE.store(Location::caller().line(), Relaxed);
        panic!();
    }

    async fn bar_rpit_in_trait() {
        LINE.store(Location::caller().line(), Relaxed);
        panic!();
    }
    fn bar_rpit_in_impl() -> impl Future<Output = ()> {
        async {
            LINE.store(Location::caller().line(), Relaxed);
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
    panic!();
}

fn foo_gen_fn() {
    let mut iter = bar_gen_fn();
    let _ = iter.next();
}

// Since compilation is expected to fail for this fn when `closure_track_caller`
// is disabled, we test that separately in `async-closure-gate.rs`
#[cfg(any(cls, afn_cls, cls_opt, afn_cls_opt))]
fn foo_gen_block() {
    let mut iter = #[track_caller]
    gen {
        LINE.store(Location::caller().line(), Relaxed);
        panic!();
        yield ();
    };
    let _ = iter.next();
}

// Since compilation is expected to fail for this fn when `closure_track_caller`
// is disabled, we test that separately in `async-closure-gate.rs`
#[cfg(any(cls, afn_cls, cls_opt, afn_cls_opt))]
fn foo_coroutine() {
    let coro = #[track_caller]
    #[coroutine]
    || {
        LINE.store(Location::caller().line(), Relaxed);
        panic!();
        yield ();
    };
    let coro = std::pin::pin!(coro);
    let _ = coro.resume(());
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
// uses the location where the future is awaited or polled.
// The correct behavior as per T-lang is to use the location where the function is called.
fn main() {
    assert_eq!(panicked_at(|| block_on(foo())), 60);
    assert_eq!(LINE.load(Relaxed), 59);

    #[cfg(any(afn, afn_cls, afn_opt, afn_cls_opt))]
    assert_eq!(panicked_at(|| block_on(foo_track_caller())), 72);
    #[cfg(any(afn, afn_cls, afn_opt, afn_cls_opt))]
    assert_eq!(LINE.load(Relaxed), 71);
    #[cfg(any(cls, nofeat, cls_opt, nofeat_opt))]
    assert_eq!(panicked_at(|| block_on(foo_track_caller())), 72);
    #[cfg(any(cls, nofeat, cls_opt, nofeat_opt))]
    assert_eq!(LINE.load(Relaxed), 71);

    #[cfg(any(afn, afn_cls, afn_opt, afn_cls_opt))]
    assert_eq!(panicked_at(|| block_on(foo_assoc())), 87);
    #[cfg(any(afn, afn_cls, afn_opt, afn_cls_opt))]
    assert_eq!(LINE.load(Relaxed), 86);
    #[cfg(any(cls, nofeat, cls_opt, nofeat_opt))]
    assert_eq!(panicked_at(|| block_on(foo_assoc())), 87);
    #[cfg(any(cls, nofeat, cls_opt, nofeat_opt))]
    assert_eq!(LINE.load(Relaxed), 86);

    #[cfg(any(afn_cls, afn_cls_opt))]
    assert_eq!(panicked_at(|| block_on(foo_closure())), 106);
    #[cfg(any(afn_cls, afn_cls_opt))]
    assert_eq!(LINE.load(Relaxed), 106);
    // FIXME(closure_track_caller): if closure_track_caller is enabled, but
    // async_fn_track_caller is disabled, then #[track_caller] on async closures
    // silently do nothing. Either it should function, or we should emit a warning.
    // See #161961
    #[cfg(any(cls, cls_opt))]
    assert_eq!(panicked_at(|| block_on(foo_closure())), 103);
    #[cfg(any(cls, cls_opt))]
    assert_eq!(LINE.load(Relaxed), 102);

    #[cfg(any(cls, afn_cls))]
    assert_eq!(panicked_at(|| block_on(foo_block())), 118);
    #[cfg(any(cls, afn_cls))]
    assert_eq!(LINE.load(Relaxed), 118);

    #[cfg(any(afn, afn_cls, afn_opt, afn_cls_opt))]
    assert_eq!(panicked_at(|| foo_manual_poll()), 125);
    #[cfg(any(afn, afn_cls, afn_opt, afn_cls_opt))]
    assert_eq!(LINE.load(Relaxed), 124);
    #[cfg(any(cls, nofeat, cls_opt, nofeat_opt))]
    assert_eq!(panicked_at(|| foo_manual_poll()), 125);
    #[cfg(any(cls, nofeat, cls_opt, nofeat_opt))]
    assert_eq!(LINE.load(Relaxed), 124);

    assert_eq!(panicked_at(|| block_on(foo_trait_attr_nowhere())), 152);
    assert_eq!(LINE.load(Relaxed), 151);

    // FIXME(async_fn_track_caller): This case just currently doesn't work.
    assert_eq!(panicked_at(|| block_on(foo_trait_attr_in_trait())), 156);
    assert_eq!(LINE.load(Relaxed), 155);

    #[cfg(any(afn, afn_cls, afn_opt, afn_cls_opt))]
    assert_eq!(panicked_at(|| block_on(foo_trait_attr_in_impl())), 162);
    #[cfg(any(afn, afn_cls, afn_opt, afn_cls_opt))]
    assert_eq!(LINE.load(Relaxed), 161);
    #[cfg(any(cls, nofeat, cls_opt, nofeat_opt))]
    assert_eq!(panicked_at(|| block_on(foo_trait_attr_in_impl())), 162);
    #[cfg(any(cls, nofeat, cls_opt, nofeat_opt))]
    assert_eq!(LINE.load(Relaxed), 161);

    #[cfg(any(afn, afn_cls, afn_opt, afn_cls_opt))]
    assert_eq!(panicked_at(|| block_on(foo_trait_attr_in_both())), 168);
    #[cfg(any(afn, afn_cls, afn_opt, afn_cls_opt))]
    assert_eq!(LINE.load(Relaxed), 167);
    #[cfg(any(cls, nofeat, cls_opt, nofeat_opt))]
    assert_eq!(panicked_at(|| block_on(foo_trait_attr_in_both())), 168);
    #[cfg(any(cls, nofeat, cls_opt, nofeat_opt))]
    assert_eq!(LINE.load(Relaxed), 167);

    // FIXME(async_fn_track_caller): This case just currently doesn't work.
    assert_eq!(panicked_at(|| block_on(foo_rpit_in_trait())), 173);
    assert_eq!(LINE.load(Relaxed), 172);

    assert_eq!(panicked_at(|| block_on(foo_rpit_in_impl())), 178);
    assert_eq!(LINE.load(Relaxed), 177);

    #[cfg(any(afn, afn_cls, afn_opt, afn_cls_opt))]
    assert_eq!(panicked_at(|| foo_gen_fn()), 212);
    #[cfg(any(afn, afn_cls, afn_opt, afn_cls_opt))]
    assert_eq!(LINE.load(Relaxed), 211);
    #[cfg(any(cls, nofeat, cls_opt, nofeat_opt))]
    assert_eq!(panicked_at(|| foo_gen_fn()), 212);
    #[cfg(any(cls, nofeat, cls_opt, nofeat_opt))]
    assert_eq!(LINE.load(Relaxed), 211);

    #[cfg(any(cls, afn_cls, cls_opt, afn_cls_opt))]
    assert_eq!(panicked_at(|| foo_gen_block()), 230);
    #[cfg(any(cls, afn_cls, cls_opt, afn_cls_opt))]
    assert_eq!(LINE.load(Relaxed), 230);

    #[cfg(any(cls, afn_cls, cls_opt, afn_cls_opt))]
    assert_eq!(panicked_at(|| foo_coroutine()), 245);
    #[cfg(any(cls, afn_cls, cls_opt, afn_cls_opt))]
    assert_eq!(LINE.load(Relaxed), 245);
}
