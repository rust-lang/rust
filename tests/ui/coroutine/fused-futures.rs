//@ run-pass
//@ edition: 2024
//@ compile-flags: -Zfused-futures

use std::pin::pin;
use std::task::{Context, Poll, Waker};

async fn foo() -> u8 {
    12
}

const N: usize = 10;

fn main() {
    let cx = &mut Context::from_waker(Waker::noop());
    let mut x = pin!(foo());
    assert_eq!(x.as_mut().poll(cx), Poll::Ready(12));
    for _ in 0..N {
        assert_eq!(x.as_mut().poll(cx), Poll::Pending);
    }
}
