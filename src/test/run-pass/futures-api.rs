// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(arbitrary_self_types, futures_api, pin)]
#![allow(unused)]

use std::boxed::PinBox;
use std::future::Future;
use std::mem::PinMut;
use std::rc::Rc;
use std::sync::{
    Arc,
    atomic::{self, AtomicUsize},
};
use std::task::{
    Context, Poll,
    Wake, Waker, LocalWaker,
    Executor, TaskObj, SpawnObjError,
    local_waker, local_waker_from_nonlocal,
};

struct Counter {
    local_wakes: AtomicUsize,
    nonlocal_wakes: AtomicUsize,
}

impl Wake for Counter {
    fn wake(this: &Arc<Self>) {
        this.nonlocal_wakes.fetch_add(1, atomic::Ordering::SeqCst);
    }

    unsafe fn wake_local(this: &Arc<Self>) {
        this.local_wakes.fetch_add(1, atomic::Ordering::SeqCst);
    }
}

struct NoopExecutor;

impl Executor for NoopExecutor {
    fn spawn_obj(&mut self, _: TaskObj) -> Result<(), SpawnObjError> {
        Ok(())
    }
}

struct MyFuture;

impl Future for MyFuture {
    type Output = ();
    fn poll(self: PinMut<Self>, cx: &mut Context) -> Poll<Self::Output> {
        // Ensure all the methods work appropriately
        cx.waker().wake();
        cx.waker().wake();
        cx.local_waker().wake();
        cx.executor().spawn_obj(PinBox::new(MyFuture).into()).unwrap();
        Poll::Ready(())
    }
}

fn test_local_waker() {
    let counter = Arc::new(Counter {
        local_wakes: AtomicUsize::new(0),
        nonlocal_wakes: AtomicUsize::new(0),
    });
    let waker = unsafe { local_waker(counter.clone()) };
    let executor = &mut NoopExecutor;
    let cx = &mut Context::new(&waker, executor);
    assert_eq!(Poll::Ready(()), PinMut::new(&mut MyFuture).poll(cx));
    assert_eq!(1, counter.local_wakes.load(atomic::Ordering::SeqCst));
    assert_eq!(2, counter.nonlocal_wakes.load(atomic::Ordering::SeqCst));
}

fn test_local_as_nonlocal_waker() {
    let counter = Arc::new(Counter {
        local_wakes: AtomicUsize::new(0),
        nonlocal_wakes: AtomicUsize::new(0),
    });
    let waker: LocalWaker = local_waker_from_nonlocal(counter.clone());
    let executor = &mut NoopExecutor;
    let cx = &mut Context::new(&waker, executor);
    assert_eq!(Poll::Ready(()), PinMut::new(&mut MyFuture).poll(cx));
    assert_eq!(0, counter.local_wakes.load(atomic::Ordering::SeqCst));
    assert_eq!(3, counter.nonlocal_wakes.load(atomic::Ordering::SeqCst));
}

fn main() {
    test_local_waker();
    test_local_as_nonlocal_waker();
}
