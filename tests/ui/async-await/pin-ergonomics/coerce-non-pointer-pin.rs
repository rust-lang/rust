//@ check-pass

#![feature(pin_ergonomics)]
//~^ WARN the feature `pin_ergonomics` is incomplete

use std::pin::Pin;

fn main() {
    let _: Pin<Box<()>> = Box::pin(());
}
