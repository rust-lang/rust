//@ run-pass
//@ check-run-results
// Test async drop of coroutine `bar` (with internal async drop),
// stopped at the middle of execution, with AsyncDrop object Foo active.

#![feature(async_drop)]
#![allow(incomplete_features)]

//@ edition: 2021

use std::mem::ManuallyDrop;

use std::{
    future::{Future, async_drop_in_place, AsyncDrop},
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
    block_on_and_drop_in_the_middle(bar(10));
    println!("done")
}

pub struct MiddleFuture {
    first_call: bool,
}
impl Future for MiddleFuture {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.first_call {
            println!("MiddleFuture first poll");
            self.first_call = false;
            Poll::Pending
        } else {
            println!("MiddleFuture Ready");
            Poll::Ready(())
        }
    }
}

async fn bar(ident_base: usize) {
    let middle = MiddleFuture { first_call: true };
    let mut _first = Foo::new(ident_base);
    middle.await; // Hanging `bar` future before Foo drop
}

fn block_on_and_drop_in_the_middle<F>(fut_unpin: F) -> F::Output
where
    F: Future<Output = ()>,
{
    let mut fut_pin = pin!(ManuallyDrop::new(fut_unpin));
    let mut fut: Pin<&mut F> = unsafe {
        Pin::map_unchecked_mut(fut_pin.as_mut(), |x| &mut **x)
    };
    let (waker, rx) = simple_waker();
    let mut context = Context::from_waker(&waker);
    let poll1 = fut.as_mut().poll(&mut context);
    assert!(poll1.is_pending());

    let drop_fut_unpin = unsafe { async_drop_in_place(fut.get_unchecked_mut()) };
    let mut drop_fut: Pin<&mut _> = pin!(drop_fut_unpin);
    loop {
        match drop_fut.as_mut().poll(&mut context) {
            Poll::Ready(()) => break,
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
