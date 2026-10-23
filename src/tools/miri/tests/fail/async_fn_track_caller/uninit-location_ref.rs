// Test when the captured `&Location` in a future is an uninitialized reference.

//@ normalize-stderr-test: "(\n)ALLOC \(.*\) \{\n(.*\n)*\}(\n)" -> "${1}ALLOC DUMP${3}"
//@ normalize-stderr-test: "\[0x[0-9a-z]..0x[0-9a-z]\]" -> "[0xX..0xY]"

#![feature(async_fn_track_caller)]

use std::future::Future;
use std::mem::MaybeUninit;
use std::panic::Location;
use std::pin::Pin;
use std::task::{Context, Waker};

#[track_caller]
async fn foo() {
    // UB when we touch the invalid captured `&Location`.
    bar();
    //~^ERROR: this operation requires initialized memory
}

#[track_caller]
fn bar() {}

fn main() {
    let mut future = foo();
    // The future has the captured caller location followed by a discriminant.
    assert_eq!(size_of_val(&future), 2 * size_of::<*const ()>());
    unsafe {
        // Make sure that the discriminant actually comes after the captured caller location.
        assert_eq!((&raw const future).cast::<u8>().add(size_of::<*const Location>()).read(), 0);
        // Deinitialize the captured caller location (`&Location`).
        (&raw mut future).cast::<MaybeUninit<*const Location>>().write(MaybeUninit::uninit());
    }
    let future = unsafe { Pin::new_unchecked(&mut future) };
    let mut cx = Context::from_waker(Waker::noop());
    let _ = future.poll(&mut cx);
}
