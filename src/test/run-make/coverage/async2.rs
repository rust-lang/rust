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

// FIXME(#83985): The auto-generated closure in an async function is failing to include
// the println!() and `let` assignment lines in the coverage code region(s), as it does in the
// non-async function above, unless the `println!()` is inside a covered block.
async fn async_func() {
    println!("async_func was covered");
    let b = true;
    if b {
        println!("async_func println in block");
    }
}

// FIXME(#83985): As above, this async function only has the `println!()` macro call, which is not
// showing coverage, so the entire async closure _appears_ uncovered; but this is not exactly true.
// It's only certain kinds of lines and/or their context that results in missing coverage.
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
