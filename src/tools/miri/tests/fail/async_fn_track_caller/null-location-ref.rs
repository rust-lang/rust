// Test when the captured `&Location` in a future is null.

#![feature(async_fn_track_caller)]

use std::future::Future;
use std::panic::Location;
use std::pin::Pin;
use std::ptr;
use std::task::{Context, Waker};

#[track_caller]
async fn foo() {
    // UB when we touch the invalid captured `&Location`.
    // FIXME: This should be UB, but isn't UB yet.
    bar();
    baz();
    //~^ERROR: constructing invalid value of type &std::panic::Location<'_>: encountered a null reference
}

#[track_caller]
fn bar() {}

#[track_caller]
fn baz() {
    let _ = Location::caller();
}

fn main() {
    let mut future = foo();
    // The future has the captured caller location followed by a discriminant.
    assert_eq!(size_of_val(&future), 2 * size_of::<*const Location>());
    unsafe {
        // Make sure that the discriminant actually comes after the captured caller location.
        assert_eq!((&raw const future).cast::<u8>().add(size_of::<*const Location>()).read(), 0);
        // Zero out the captured caller location (`&Location`).
        (&raw mut future).cast::<*const Location>().write(ptr::null());
    }
    let future = unsafe { Pin::new_unchecked(&mut future) };
    let mut cx = Context::from_waker(Waker::noop());
    let _ = future.poll(&mut cx);
}
