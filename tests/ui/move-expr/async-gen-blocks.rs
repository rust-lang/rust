//@ edition: 2024
//@ run-pass
#![allow(incomplete_features)]
#![feature(async_iterator, gen_blocks, move_expr)]

use std::async_iter::AsyncIterator;
use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

struct PendingOnce {
    pending: bool,
}

impl PendingOnce {
    fn new() -> Self {
        Self { pending: true }
    }
}

impl Future for PendingOnce {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.pending {
            self.pending = false;
            cx.waker().wake_by_ref();
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }
}

fn poll_next<I: AsyncIterator>(iter: Pin<&mut I>) -> Poll<Option<I::Item>> {
    let cx = &mut Context::from_waker(Waker::noop());
    AsyncIterator::poll_next(iter, cx)
}

fn ready_next<I: AsyncIterator>(iter: Pin<&mut I>) -> Option<I::Item> {
    match poll_next(iter) {
        Poll::Ready(item) => item,
        Poll::Pending => panic!("async iterator unexpectedly returned pending"),
    }
}

fn main() {
    let created = Cell::new(0);
    let mut iter = Box::pin(async gen {
        let n = move({
            created.set(created.get() + 1);
            created.get()
        });
        yield n;
        yield n + 1;
    });
    assert_eq!(created.get(), 1);
    assert_eq!(ready_next(iter.as_mut()), Some(1));
    assert_eq!(ready_next(iter.as_mut()), Some(2));
    assert_eq!(ready_next(iter.as_mut()), None);

    let x = Arc::new(String::from("hello"));
    assert_eq!(Arc::strong_count(&x), 1);
    let mut iter = Box::pin(async gen {
        let value = move(x.clone());
        yield Arc::strong_count(&value);
        PendingOnce::new().await;
        yield Arc::strong_count(&value);
    });
    assert_eq!(Arc::strong_count(&x), 2);
    assert_eq!(ready_next(iter.as_mut()), Some(2));
    assert!(matches!(poll_next(iter.as_mut()), Poll::Pending));
    assert_eq!(ready_next(iter.as_mut()), Some(2));
    drop(iter);
    assert_eq!(Arc::strong_count(&x), 1);

    let y = Arc::new(String::from("nested"));
    assert_eq!(Arc::strong_count(&y), 1);
    let mut iter = Box::pin(async gen {
        let value = move(move(y.clone()));
        yield Arc::strong_count(&value);
    });
    assert_eq!(Arc::strong_count(&y), 2);
    assert_eq!(ready_next(iter.as_mut()), Some(2));
    drop(iter);
    assert_eq!(Arc::strong_count(&y), 1);

    let z = Arc::new(String::from("async gen move"));
    assert_eq!(Arc::strong_count(&z), 1);
    let mut iter = Box::pin(async gen move {
        let value = move(z.clone());
        yield Arc::strong_count(&value);
    });
    assert_eq!(Arc::strong_count(&z), 2);
    assert_eq!(ready_next(iter.as_mut()), Some(2));
    drop(iter);
    assert_eq!(Arc::strong_count(&z), 1);
}
