//@ run-pass
//@ check-run-results
//@ edition: 2024

#![feature(async_drop)]
#![allow(incomplete_features)]

use std::{future::Future, pin::pin, task::{Context, Poll, Waker}};

struct Bar;
impl Drop for Bar {
    fn drop(&mut self) { println!("Bar dropped"); }
}

async fn outer() {
    let b = Bar;
    let _inner = async move { let _x = &b; };
    println!("end of outer");
}

fn main() {
    let mut f = pin!(outer());
    let mut cx = Context::from_waker(Waker::noop());
    while let Poll::Pending = f.as_mut().poll(&mut cx) {}
    println!("done");
}
