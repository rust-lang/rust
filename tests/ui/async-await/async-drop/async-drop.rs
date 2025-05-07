//@ run-pass
//@ check-run-results
// struct `Foo` has both sync and async drop.
// Sync version is called in sync context, async version is called in async function.

#![feature(async_drop)]
#![allow(incomplete_features)]

use std::mem::ManuallyDrop;

//@ edition: 2021

#[inline(never)]
fn myprintln(msg: &str, my_resource_handle: usize) {
    println!("{} : {}", msg, my_resource_handle);
}

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
        myprintln("Foo::new()", my_resource_handle);
        out
    }
}

impl Drop for Foo {
    fn drop(&mut self) {
        myprintln("Foo::drop()", self.my_resource_handle);
    }
}

impl AsyncDrop for Foo {
    async fn drop(self: Pin<&mut Self>) {
        myprintln("Foo::async drop()", self.my_resource_handle);
    }
}

fn main() {
    {
        let _ = Foo::new(7);
    }
    println!("Middle");
    block_on(bar(10));
    println!("Done")
}

async fn bar(ident_base: usize) {
    let mut _first = Foo::new(ident_base);
}

fn block_on<F>(fut_unpin: F) -> F::Output
where
    F: Future,
{
    let mut fut_pin = pin!(ManuallyDrop::new(fut_unpin));
    let mut fut: Pin<&mut F> = unsafe {
        Pin::map_unchecked_mut(fut_pin.as_mut(), |x| &mut **x)
    };
    let (waker, rx) = simple_waker();
    let mut context = Context::from_waker(&waker);
    let rv = loop {
        match fut.as_mut().poll(&mut context) {
            Poll::Ready(out) => break out,
            // expect wake in polls
            Poll::Pending => rx.try_recv().unwrap(),
        }
    };
    let drop_fut_unpin = unsafe { async_drop_in_place(fut.get_unchecked_mut()) };
    let mut drop_fut: Pin<&mut _> = pin!(drop_fut_unpin);
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
