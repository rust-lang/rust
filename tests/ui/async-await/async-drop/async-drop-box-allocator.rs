//@ run-pass
//@ check-run-results
// struct `Foo` has both sync and async drop.
// It's used as the allocator of a `Box` which is conditionally moved out of.
// Sync version is called in sync context, async version is called in async function.

#![feature(async_drop, allocator_api)]
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
    alloc::{AllocError, Allocator, Global, Layout},
    ptr::NonNull,
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

unsafe impl Allocator for Foo {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        Global.deallocate(ptr, layout);
    }
}

struct HasDrop;
impl Drop for HasDrop {
    fn drop(&mut self) {}
}

fn main() {
    {
        let b = Box::new_in(HasDrop, Foo::new(7));

        if true {
            let _x = *b;
        } else {
            let _y = b;
        }
    }
    println!("Middle");
    block_on(bar(10));
    println!("Done")
}

async fn bar(ident_base: usize) {
    let b = Box::new_in(HasDrop, Foo::new(ident_base));

    if true {
        let _x = *b;
    } else {
        let _y = b;
    }
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
