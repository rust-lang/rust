//! Regression test for <https://github.com/rust-lang/rust/issues/153438>: when there's a type
//! expectation on `pin!`'s result, make sure we don't deref-coerce the argument to
//! `Pin::new_unchecked` to get its type to match up. That violates the pinning invariant, leading
//! to unsoundness!
//@ check-pass

use std::pin::{Pin, pin};

fn wrong_pin<T>(data: &mut T, callback: impl FnOnce(Pin<&mut T>)) {
    callback(pin!(data));
}

fn main() {}
