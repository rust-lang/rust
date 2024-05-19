//@ run-pass
//@ edition:2021
//@ check-run-results

#![feature(dyn_star)]
//~^ WARN the feature `dyn_star` is incomplete and may not be safe to use and/or cause compiler crashes
#![feature(noop_waker)]

use std::future::Future;

async fn foo(f: dyn* Future<Output = i32>) {
    println!("value: {}", f.await);
}

async fn async_main() {
    foo(Box::pin(async { 1 })).await
}

// ------------------------------------------------------------------------- //
// Implementation Details Below...

use std::pin::pin;
use std::task::*;

fn main() {
    let mut fut = pin!(async_main());

    // Poll loop, just to test the future...
    let ctx = &mut Context::from_waker(Waker::noop());

    loop {
        match fut.as_mut().poll(ctx) {
            Poll::Pending => {}
            Poll::Ready(()) => break,
        }
    }
}
