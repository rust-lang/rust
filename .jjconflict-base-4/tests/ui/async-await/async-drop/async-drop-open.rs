//@ run-pass
//@ check-run-results
// struct `Foo` has both sync and async drop.
// Struct `Complex` contains three `Foo` fields and one of them is moved out.

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

#[allow(dead_code)]
struct Complex {
    field1: Foo,
    field2: Foo,
    field3: Foo,
}

impl Complex {
    fn new(my_resource_handle: usize) -> Self {
        myprintln("Complex::new()", my_resource_handle);
        let field1 = Foo::new(my_resource_handle);
        let field2 = Foo::new(my_resource_handle + 1);
        let field3 = Foo::new(my_resource_handle + 2);
        Complex { field1, field2, field3 }
    }
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
    // Inside field1 and field3 of Complex must be dropped (as async drop)
    // field2 must be dropped here (as sync drop)
    {
        let _field2 = block_on(bar(10));
    }
    println!("Done")
}

async fn bar(ident_base: usize) -> Foo {
    let complex = Complex::new(ident_base);
    complex.field2
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
