#![feature(coverage_attribute)]
//@ edition: 2021

use core::future::Future;
use core::pin::pin;
use core::task::{Context, Poll, Waker};

/// Dummy "executor" that just repeatedly polls a future until it's ready.
#[coverage(off)]
pub fn block_on<F: Future>(mut future: F) -> F::Output {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());

    loop {
        if let Poll::Ready(val) = future.as_mut().poll(&mut context) {
            break val;
        }
    }
}
