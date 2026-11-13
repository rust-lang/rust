//@ known-bug: #154782
//@ edition: 2024
#![feature(pin_ergonomics)]
use core::pin::Pin;
fn test_idempotency<T: Future>(x: Pin<&mut T>) {
    || {
        x.poll(loop {});
    };
}
