// aux-build:arc_wake.rs

extern crate arc_wake;

use std::future::Future;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{self, AtomicUsize},
};
use std::task::{
    Context, Poll,
};
use arc_wake::ArcWake;

struct Counter {
    wakes: AtomicUsize,
}

impl ArcWake for Counter {
    fn wake(self: Arc<Self>) {
        Self::wake_by_ref(&self)
    }
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.wakes.fetch_add(1, atomic::Ordering::SeqCst);
    }
}

struct MyFuture;

impl Future for MyFuture {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Wake twice
        let waker = cx.waker();
        waker.wake_by_ref();
        waker.wake_by_ref();
        Poll::Ready(())
    }
}

fn test_waker() {
    let counter = Arc::new(Counter {
        wakes: AtomicUsize::new(0),
    });
    let waker = ArcWake::into_waker(counter.clone());
    assert_eq!(2, Arc::strong_count(&counter));
    {
        let mut context = Context::from_waker(&waker);
        assert_eq!(Poll::Ready(()), Pin::new(&mut MyFuture).poll(&mut context));
        assert_eq!(2, counter.wakes.load(atomic::Ordering::SeqCst));
    }
    drop(waker);
    assert_eq!(1, Arc::strong_count(&counter));
}

fn main() {
    test_waker();
}
