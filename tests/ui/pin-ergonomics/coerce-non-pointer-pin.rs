//@ check-pass

#![feature(pin_ergonomics)]

use std::pin::Pin;

fn main() {
    let _: Pin<Box<()>> = Box::pin(());
}
