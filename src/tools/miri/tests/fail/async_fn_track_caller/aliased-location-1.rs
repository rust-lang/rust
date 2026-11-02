// Test when the captured `&Location` in a future is aliased with a `&mut`

//@compile-flags: -Zmiri-tree-borrows
//@normalize-stderr-test: "\[0x[a-fx\d.]+\]" -> "[RANGE]"

#![feature(async_fn_track_caller)]

use std::future::{Future, poll_fn};
use std::panic::Location;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

// Creates a future whose poll() returns Pending once, then returns Ready
fn pending_once() -> impl Future<Output = ()> {
    let mut is_ready = false;
    poll_fn(move |_| {
        if is_ready {
            Poll::Ready(())
        } else {
            is_ready = true;
            Poll::Pending
        }
    })
}

#[track_caller]
async fn foo() {
    pending_once().await;
    let location: &Location = Location::caller();
    let _location = *location;
    pending_once().await;
    let _location = location;
    let _location = *location;
}

fn main() {
    let mut future = foo();
    // Initialize a `Location` with some dummy value.
    let mut fake_caller_location = *Location::caller();
    // The future has the captured caller location followed by some data.
    assert_eq!(size_of_val(&future), 4 * size_of::<*const ()>());
    unsafe {
        // Make the captured caller location (`&Location`) point to memory we control.
        (&raw mut future).cast::<*const Location>().write(&raw const fake_caller_location);
    }
    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    let mut cx = Context::from_waker(Waker::noop());

    let l = &mut fake_caller_location;
    *l = *Location::caller();
    // Run the future to the first pending_once().
    assert_eq!(future.as_mut().poll(&mut cx), Poll::Pending);
    *l = *Location::caller();
    // Run the future to the second pending_once().
    assert_eq!(future.as_mut().poll(&mut cx), Poll::Pending);
    *l = *Location::caller();
    //~^ERROR: is forbidden
    // FIXME: Check later if this makes sense.
    // Run the future to UB.
    let _ = future.as_mut().poll(&mut cx);
    unreachable!();
}
