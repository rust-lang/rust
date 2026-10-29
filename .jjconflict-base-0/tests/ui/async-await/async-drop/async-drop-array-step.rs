//@ run-pass
//@ check-run-results
//@ edition: 2021

#![feature(async_drop)]
#![allow(incomplete_features)]

use std::future::{async_drop_in_place, AsyncDrop, Future};
use std::mem::ManuallyDrop;
use std::pin::{pin, Pin};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::task::{Context, Poll, Wake, Waker};

static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

struct YieldOnce(bool);

impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if !self.0 {
            self.0 = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }
}

struct Item(usize);

impl Drop for Item {
    fn drop(&mut self) {}
}

impl AsyncDrop for Item {
    async fn drop(self: Pin<&mut Self>) {
        let count = DROP_COUNT.fetch_add(1, Ordering::Relaxed);
        if count >= 8 {
            panic!("Infinite loop detected: array drop index reset across yield points!");
        }
        YieldOnce(false).await;
        println!("Dropping {}", self.0);
    }
}

fn block_on<F: Future>(fut_unpin: F) -> F::Output {
    let mut fut_pin = pin!(ManuallyDrop::new(fut_unpin));
    let mut fut: Pin<&mut F> = unsafe {
        Pin::map_unchecked_mut(fut_pin.as_mut(), |x| &mut **x)
    };
    let (waker, rx) = simple_waker();
    let mut context = Context::from_waker(&waker);
    let rv = loop {
        match fut.as_mut().poll(&mut context) {
            Poll::Ready(out) => break out,
            Poll::Pending => rx.try_recv().unwrap(),
        }
    };
    let drop_fut_unpin = unsafe { async_drop_in_place(fut.get_unchecked_mut()) };
    let mut drop_fut = pin!(drop_fut_unpin);
    loop {
        match drop_fut.as_mut().poll(&mut context) {
            Poll::Ready(()) => break,
            Poll::Pending => rx.try_recv().unwrap(),
        }
    }
    rv
}

fn simple_waker() -> (Waker, mpsc::Receiver<()>) {
    struct SimpleWaker {
        tx: mpsc::Sender<()>,
    }

    impl Wake for SimpleWaker {
        fn wake(self: Arc<Self>) {
            self.tx.send(()).unwrap();
        }
    }

    let (tx, rx) = mpsc::channel();
    (Waker::from(Arc::new(SimpleWaker { tx })), rx)
}

async fn run() {
    let _arr: [Item; 4] = [Item(0), Item(1), Item(2), Item(3)];
}

fn main() {
    block_on(run());
}
