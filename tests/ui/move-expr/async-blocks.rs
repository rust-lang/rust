//@ edition: 2021
//@ run-pass
#![allow(incomplete_features)]
#![feature(move_expr)]

use std::cell::Cell;
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = Box::pin(future);
    let cx = &mut Context::from_waker(Waker::noop());
    loop {
        match future.as_mut().poll(cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => {}
        }
    }
}

fn main() {
    let created = Cell::new(0);
    let fut = async {
        let n = move({
            created.set(created.get() + 1);
            created.get()
        });
        n
    };
    assert_eq!(created.get(), 1);
    drop(fut);

    let x = Arc::new(String::from("hello"));
    assert_eq!(Arc::strong_count(&x), 1);
    let fut = async { move(x.clone()) };
    assert_eq!(Arc::strong_count(&x), 2);
    drop(fut);
    assert_eq!(Arc::strong_count(&x), 1);

    let y = Arc::new(String::from("nested once"));
    let weak = Arc::downgrade(&y);
    let fut = async {
        let inner = async {
            drop(move(y.clone()));
        };
        assert_eq!(weak.strong_count(), 2);
        inner.await;
        assert_eq!(weak.strong_count(), 1);
        drop(y);
    };
    assert_eq!(weak.strong_count(), 1);
    block_on(fut);
    assert_eq!(weak.strong_count(), 0);

    let y = Arc::new(String::from("nested twice"));
    let weak = Arc::downgrade(&y);
    let fut = async {
        let inner = async {
            drop(move(move(y.clone())));
        };
        assert_eq!(weak.strong_count(), 2);
        inner.await;
        assert_eq!(weak.strong_count(), 1);
    };
    assert_eq!(weak.strong_count(), 2);
    block_on(fut);
    assert_eq!(weak.strong_count(), 1);
    assert_eq!(&*y, "nested twice");

    let z = Arc::new(String::from("async move"));
    assert_eq!(Arc::strong_count(&z), 1);
    let fut = async move { move(z.clone()) };
    assert_eq!(Arc::strong_count(&z), 2);
    drop(fut);
    assert_eq!(Arc::strong_count(&z), 1);
}
