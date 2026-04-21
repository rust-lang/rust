//@ run-pass
//@ edition: 2024

#![feature(fused_attribute)]

use std::pin::pin;
use std::task::{Context, Poll, Waker};

#[fused]
async fn fused() -> &'static str {
    "done"
}

fn main() {
    let mut fut = pin!(fused());
    let cx = &mut Context::from_waker(Waker::noop());
    assert_eq!(fut.as_mut().poll(cx), Poll::Ready("done"));
    for _ in 0..10 {
        assert_eq!(fut.as_mut().poll(cx), Poll::Pending);
    }
}
