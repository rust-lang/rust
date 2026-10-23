// Test when the captured `&Location` in a future is aliased with a `&mut`

//@compile-flags: -Zmiri-tree-borrows

#![feature(async_fn_track_caller)]

use std::future::Future;
use std::panic::Location;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

#[track_caller]
async fn foo() {}

fn main() {
    let mut future = foo();
    // Initialize a `Location` with some dummy value.
    let mut fake_caller_location = *Location::caller();
    // The future has the captured caller location followed by a discriminant.
    assert_eq!(size_of_val(&future), 2 * size_of::<*const ()>());
    unsafe {
        // Make the captured caller location (`&Location`) point to memory we control.
        (&raw mut future).cast::<*const Location>().write(&raw const fake_caller_location);
    }
    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    let mut cx = Context::from_waker(Waker::noop());

    let l = &mut fake_caller_location;
    *l = *Location::caller();
    // Run the future to completion.
    assert_eq!(future.as_mut().poll(&mut cx), Poll::Ready(()));
    *l = *Location::caller();
    //~^ERROR: write access through
}
