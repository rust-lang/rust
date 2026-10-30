// Test when the captured `&Location` in a future points to an uninitialized `Location`.

//@ normalize-stderr-test: "(\n)ALLOC \(.*\) \{\n(.*\n)*\}(\n)" -> "${1}ALLOC DUMP${3}"
//@ normalize-stderr-test: "\[0x[0-9a-z]..0x[0-9a-z]\]" -> "[0xX..0xY]"
//@error-in-other-file: this operation requires initialized memory

#![feature(async_fn_track_caller)]

use std::future::Future;
use std::mem::MaybeUninit;
use std::panic::Location;
use std::pin::Pin;
use std::task::{Context, Waker};

#[track_caller]
async fn foo() {
    // Passing the captured caller location as an implicit argument is fine.
    bar();
}

#[track_caller]
fn bar() {
    // Explicitly grabbing the caller location is fine.
    let _ = Location::caller();
    // Panicking is not fine, since it actually reads the `Location`
    panic!();
}

fn main() {
    let mut future = foo();
    // The future has the captured caller location followed by a discriminant.
    assert_eq!(size_of_val(&future), 2 * size_of::<*const ()>());
    let fake_caller_location = MaybeUninit::<Location>::uninit();
    unsafe {
        // Make sure that the discriminant actually comes after the captured caller location.
        assert_eq!((&raw const future).cast::<u8>().add(size_of::<*const ()>()).read(), 0);
        // Make the captured caller location (`&Location`) point to uninitialized memory.
        (&raw mut future).cast::<*const Location>().write(fake_caller_location.as_ptr());
    }
    let future = unsafe { Pin::new_unchecked(&mut future) };
    let mut cx = Context::from_waker(Waker::noop());
    let _ = future.poll(&mut cx);
}
