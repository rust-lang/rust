// Test that we don't store uninitialized locals in futures from `async fn`.
//
// The exact sizes can change by a few bytes (we'd like to know when they do).
// What we don't want to see is the wrong multiple of 1024 (the size of `Big`)
// being reflected in the size.

//@ ignore-emscripten (sizes don't match)
//@ needs-unwind Size of Futures change on panic=abort
//@ run-pass

//@ edition:2018

#![allow(unused_variables, unused_assignments)]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

const BIG_FUT_SIZE: usize = 1024;
struct Big(#[allow(dead_code)] [u8; BIG_FUT_SIZE]);

impl Big {
    fn new() -> Self {
        Big([0; BIG_FUT_SIZE])
    }
}

impl Drop for Big {
    fn drop(&mut self) {}
}

#[allow(dead_code)]
struct Joiner {
    a: Option<Big>,
    b: Option<Big>,
    c: Option<Big>,
}

impl Future for Joiner {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _ctx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(())
    }
}

fn noop() {}
async fn fut() {}

async fn single() {
    let x;
    fut().await;
    x = Big::new();
}

async fn single_with_noop() {
    let x;
    fut().await;
    noop();
    x = Big::new();
    noop();
}

async fn joined() {
    let joiner;
    let a = Big::new();
    let b = Big::new();
    let c = Big::new();

    fut().await;
    joiner = Joiner { a: Some(a), b: Some(b), c: Some(c) };
}

async fn joined_with_noop() {
    let joiner;
    let a = Big::new();
    let b = Big::new();
    let c = Big::new();

    fut().await;
    noop();
    joiner = Joiner { a: Some(a), b: Some(b), c: Some(c) };
    noop();
}

async fn join_retval() -> Joiner {
    let a = Big::new();
    let b = Big::new();
    let c = Big::new();

    fut().await;
    noop();
    Joiner { a: Some(a), b: Some(b), c: Some(c) }
}

fn main() {
    assert_eq!(2, std::mem::size_of_val(&single()));
    assert_eq!(3, std::mem::size_of_val(&single_with_noop()));
    assert_eq!(3074, std::mem::size_of_val(&joined()));
    assert_eq!(3078, std::mem::size_of_val(&joined_with_noop()));
    assert_eq!(3074, std::mem::size_of_val(&join_retval()));
}
