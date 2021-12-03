// Test that we don't duplicate storage for futures moved around in .await, and
// for futures moved into other futures.
//
// The exact sizes can change by a few bytes (we'd like to know when they do).
// What we don't want to see is the wrong multiple of 1024 (the size of BigFut)
// being reflected in the size.
//
// See issue #59123 for a full explanation.

// ignore-emscripten (sizes don't match)
// run-pass

// edition:2018

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

const BIG_FUT_SIZE: usize = 1024;
struct BigFut([u8; BIG_FUT_SIZE]);

impl BigFut {
    fn new() -> Self {
        BigFut([0; BIG_FUT_SIZE])
    }
}

impl Drop for BigFut {
    fn drop(&mut self) {}
}

impl Future for BigFut {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _ctx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(())
    }
}

#[allow(dead_code)]
struct Joiner {
    a: Option<BigFut>,
    b: Option<BigFut>,
    c: Option<BigFut>,
}

impl Future for Joiner {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _ctx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(())
    }
}

fn noop() {}

async fn single() {
    let x = BigFut::new();
    x.await;
}

async fn single_with_noop() {
    let x = BigFut::new();
    noop();
    x.await;
}

async fn joined() {
    let a = BigFut::new();
    let b = BigFut::new();
    let c = BigFut::new();

    let joiner = Joiner {
        a: Some(a),
        b: Some(b),
        c: Some(c),
    };
    joiner.await
}

async fn joined_with_noop() {
    let a = BigFut::new();
    let b = BigFut::new();
    let c = BigFut::new();

    let joiner = Joiner {
        a: Some(a),
        b: Some(b),
        c: Some(c),
    };
    noop();
    joiner.await
}

async fn mixed_sizes() {
    let a = BigFut::new();
    let b = BigFut::new();
    let c = BigFut::new();
    let d = BigFut::new();
    let e = BigFut::new();
    let joiner = Joiner {
        a: Some(a),
        b: Some(b),
        c: Some(c),
    };

    d.await;
    e.await;
    joiner.await;
}

fn main() {
    assert_eq!(1025, std::mem::size_of_val(&single()));
    assert_eq!(1026, std::mem::size_of_val(&single_with_noop()));
    assert_eq!(3076, std::mem::size_of_val(&joined()));
    assert_eq!(3076, std::mem::size_of_val(&joined_with_noop()));
    assert_eq!(6157, std::mem::size_of_val(&mixed_sizes()));
}
