//@ run-pass
//@ check-run-results
// Future `bar` with internal async drop `Foo` will have async drop itself.
// And we trying to drop this future in sync context (`block_on` func)

#![feature(async_drop)]
#![allow(incomplete_features)]

//@ edition: 2021

use std::{
    future::{Future, AsyncDrop},
    pin::{pin, Pin},
    sync::{mpsc, Arc},
    task::{Context, Poll, Wake, Waker},
};

struct Foo {
    my_resource_handle: usize,
}

impl Foo {
    fn new(my_resource_handle: usize) -> Self {
        let out = Foo {
            my_resource_handle,
        };
        println!("Foo::new({})", my_resource_handle);
        out
    }
}

impl Drop for Foo {
    fn drop(&mut self) {
        println!("Foo::drop({})", self.my_resource_handle);
    }
}

impl AsyncDrop for Foo {
    async fn drop(self: Pin<&mut Self>) {
        println!("Foo::async drop({})", self.my_resource_handle);
    }
}

fn main() {
    block_on(bar(10));
    println!("done")
}

async fn bar(ident_base: usize) {
    let mut _first = Foo::new(ident_base);
}

fn block_on<F>(fut: F) -> F::Output
where
    F: Future,
{
    let mut fut = pin!(fut);
    let (waker, rx) = simple_waker();
    let mut context = Context::from_waker(&waker);
    loop {
        match fut.as_mut().poll(&mut context) {
            Poll::Ready(out) => break out,
            // expect wake in polls
            Poll::Pending => rx.try_recv().unwrap(),
        }
    }
}

fn simple_waker() -> (Waker, mpsc::Receiver<()>) {
    struct SimpleWaker {
        tx: std::sync::mpsc::Sender<()>,
    }

    impl Wake for SimpleWaker {
        fn wake(self: Arc<Self>) {
            self.tx.send(()).unwrap();
        }
    }

    let (tx, rx) = mpsc::channel();
    (Waker::from(Arc::new(SimpleWaker { tx })), rx)
}
