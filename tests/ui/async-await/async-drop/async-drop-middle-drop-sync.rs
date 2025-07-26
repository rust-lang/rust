//@ run-pass
//@ check-run-results
// Test async drop of coroutine `bar` (with internal async drop),
// stopped at the middle of execution, with AsyncDrop object Foo active.

//#![feature(async_drop, async_drop_lib)]
#![allow(incomplete_features)]

//@ edition: 2021

use std::{
    future::Future,
    pin::{pin, Pin},
    task::{Context, Poll, Waker},
};

fn main() {
    bar(10);
    println!("done")
}

pub struct MiddleFuture {
    first_call: bool,
}
impl Drop for MiddleFuture {
    fn drop(&mut self) {
        println!("MiddleFuture::drop()");
    }
}
impl MiddleFuture {
    fn create() -> Box<dyn Future<Output = ()> + Unpin> {
        Box::new(MiddleFuture { first_call: true })
    }
}

impl Future for MiddleFuture {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.first_call {
            println!("MiddleFuture first poll");
            self.first_call = false;
            Poll::Pending
        } else {
            println!("MiddleFuture Ready");
            Poll::Ready(())
        }
    }
}

fn bar(_ident_base: usize) {
    let middle = MiddleFuture::create();
    let waker = Waker::noop();
    let mut context = Context::from_waker(&waker);
    let mut fut = pin!(middle);
    let poll1 = fut.as_mut().poll(&mut context);
    assert!(poll1.is_pending());
}
