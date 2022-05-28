// edition:2018
#![forbid(incomplete_features, unsafe_code)]
#![feature(unsafe_pin_internals)]
//~^ ERROR the feature `unsafe_pin_internals` is incomplete and may not be safe to use

use core::{marker::PhantomPinned, pin::Pin};

/// The `unsafe_pin_internals` is indeed unsound.
fn non_unsafe_pin_new_unchecked<T>(pointer: &mut T) -> Pin<&mut T> {
    Pin { pointer }
}

fn main() {
    let mut self_referential = PhantomPinned;
    let _: Pin<&mut PhantomPinned> = non_unsafe_pin_new_unchecked(&mut self_referential);
    core::mem::forget(self_referential); // move and disable drop glue!
}
