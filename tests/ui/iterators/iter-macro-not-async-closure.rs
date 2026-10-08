//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] known-bug: trait-system-refactor-initiative#302
//@[next] check-pass

// This test ensures iterators created with the `iter!` macro are not
// accidentally async closures.
//
//@ edition: 2024
//@ remap-src-base

#![feature(yield_expr, iter_macro)]

use std::task::{Waker, Context};
use std::iter::iter;
use std::pin::pin;
use std::future::Future;

async fn call_async_once(f: impl AsyncFnOnce()) {
    f().await
}

fn main() {
    let f = iter! { move || {
        for i in 0..10 {
            yield i;
        }
    }};

    let x = pin!(call_async_once(f));
    //[current]~^ ERROR AsyncFnOnce()` is not satisfied
    //[current]~| ERROR AsyncFnOnce()` is not satisfied
    x.poll(&mut Context::from_waker(Waker::noop()));
    //[current]~^ ERROR AsyncFnOnce()` is not satisfied
}
