// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --edition=2018

#![feature(arbitrary_self_types, async_await, await_macro, futures_api, pin)]

use std::boxed::PinBox;
use std::mem::PinMut;
use std::future::Future;
use std::sync::{
    Arc,
    atomic::{self, AtomicUsize},
};
use std::task::{
    Context, Poll, Wake,
    Executor, TaskObj, SpawnObjError,
    local_waker_from_nonlocal,
};

struct Counter {
    wakes: AtomicUsize,
}

impl Wake for Counter {
    fn wake(this: &Arc<Self>) {
        this.wakes.fetch_add(1, atomic::Ordering::SeqCst);
    }
}

struct NoopExecutor;
impl Executor for NoopExecutor {
    fn spawn_obj(&mut self, _: TaskObj) -> Result<(), SpawnObjError> {
        Ok(())
    }
}

struct WakeOnceThenComplete(bool);

fn wake_and_yield_once() -> WakeOnceThenComplete { WakeOnceThenComplete(false) }

impl Future for WakeOnceThenComplete {
    type Output = ();
    fn poll(mut self: PinMut<Self>, cx: &mut Context) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            cx.waker().wake();
            self.0 = true;
            Poll::Pending
        }
    }
}

fn async_block(x: u8) -> impl Future<Output = u8> {
    async move {
        await!(wake_and_yield_once());
        x
    }
}

fn async_nonmove_block(x: u8) -> impl Future<Output = u8> {
    async move {
        let future = async {
            await!(wake_and_yield_once());
            x
        };
        await!(future)
    }
}

fn async_closure(x: u8) -> impl Future<Output = u8> {
    (async move |x: u8| -> u8 {
        await!(wake_and_yield_once());
        x
    })(x)
}

async fn async_fn(x: u8) -> u8 {
    await!(wake_and_yield_once());
    x
}

async fn async_fn_with_borrow(x: &u8) -> u8 {
    await!(wake_and_yield_once());
    *x
}

fn async_fn_with_internal_borrow(y: u8) -> impl Future<Output = u8> {
    async move {
        await!(async_fn_with_borrow(&y))
    }
}

unsafe async fn unsafe_async_fn(x: u8) -> u8 {
    await!(wake_and_yield_once());
    x
}

struct Foo;

trait Bar {
    fn foo() {}
}

impl Foo {
    async fn async_method(x: u8) -> u8 {
        unsafe {
            await!(unsafe_async_fn(x))
        }
    }
}

fn test_future_yields_once_then_returns<F, Fut>(f: F)
where
    F: FnOnce(u8) -> Fut,
    Fut: Future<Output = u8>,
{
    let mut fut = PinBox::new(f(9));
    let counter = Arc::new(Counter { wakes: AtomicUsize::new(0) });
    let waker = local_waker_from_nonlocal(counter.clone());
    let executor = &mut NoopExecutor;
    let cx = &mut Context::new(&waker, executor);

    assert_eq!(0, counter.wakes.load(atomic::Ordering::SeqCst));
    assert_eq!(Poll::Pending, fut.as_pin_mut().poll(cx));
    assert_eq!(1, counter.wakes.load(atomic::Ordering::SeqCst));
    assert_eq!(Poll::Ready(9), fut.as_pin_mut().poll(cx));
}

fn main() {
    macro_rules! test {
        ($($fn_name:ident,)*) => { $(
            test_future_yields_once_then_returns($fn_name);
        )* }
    }

    test! {
        async_block,
        async_nonmove_block,
        async_closure,
        async_fn,
        async_fn_with_internal_borrow,
    }
}
