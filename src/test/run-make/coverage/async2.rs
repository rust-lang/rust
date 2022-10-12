// compile-flags: --edition=2018

use core::{
    future::Future,
    marker::Send,
    pin::Pin,
};

fn non_async_func() {
    println!("non_async_func was covered");
    let b = true;
    if b {
        println!("non_async_func println in block");
    }
}




async fn async_func() {
    println!("async_func was covered");
    let b = true;
    if b {
        println!("async_func println in block");
    }
}




async fn async_func_just_println() {
    println!("async_func_just_println was covered");
}

fn main() {
    println!("codecovsample::main");

    non_async_func();

    executor::block_on(async_func());
    executor::block_on(async_func_just_println());
}

mod executor {
    use core::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
    };

    pub fn block_on<F: Future>(mut future: F) -> F::Output {
        let mut future = unsafe { Pin::new_unchecked(&mut future) };
        use std::hint::unreachable_unchecked;
        static VTABLE: RawWakerVTable = RawWakerVTable::new(
            |_| unsafe { unreachable_unchecked() }, // clone
            |_| unsafe { unreachable_unchecked() }, // wake
            |_| unsafe { unreachable_unchecked() }, // wake_by_ref
            |_| (),
        );
        let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) };
        let mut context = Context::from_waker(&waker);

        loop {
            if let Poll::Ready(val) = future.as_mut().poll(&mut context) {
                break val;
            }
        }
    }
}
