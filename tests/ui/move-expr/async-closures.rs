//@ edition: 2021
//@ run-pass
#![allow(incomplete_features)]
#![feature(move_expr)]

use std::cell::Cell;
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

fn block_on<T>(future: impl Future<Output = T>) -> T {
    let mut future = pin!(future);
    let context = &mut Context::from_waker(Waker::noop());

    loop {
        match future.as_mut().poll(context) {
            Poll::Ready(value) => return value,
            Poll::Pending => {}
        }
    }
}

async fn call_once<T>(closure: impl AsyncFnOnce() -> T) -> T {
    closure().await
}

fn main() {
    let created = Cell::new(0);
    let c = async || {
        let n = move({
            created.set(created.get() + 1);
            created.get()
        });
        n
    };
    assert_eq!(created.get(), 1);
    assert_eq!(block_on(c()), 1);
    assert_eq!(block_on(c()), 1);
    assert_eq!(created.get(), 1);

    let x = Arc::new(String::from("hello"));
    assert_eq!(Arc::strong_count(&x), 1);

    let c = async || move(x.clone());
    assert_eq!(Arc::strong_count(&x), 2);
    let fut = c();
    assert_eq!(Arc::strong_count(&x), 2);
    drop(fut);
    assert_eq!(Arc::strong_count(&x), 1);

    let a = String::from("a");
    let b = String::from("bbb");
    let c = async || {
        let moved = move(a.clone());
        (moved, b.len())
    };
    assert_eq!(block_on(call_once(c)), (String::from("a"), 3));
}
