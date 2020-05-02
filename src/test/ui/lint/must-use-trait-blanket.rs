//! Tests the `MustUse` trait implemented with a conditional blanket impl.

#![feature(must_use_trait)]
#![deny(unused_must_use)] //~ NOTE the lint level is defined here

use std::marker::MustUse;

struct Box<T>(T);

impl<T> Box<T> {
    fn new(t: T) -> Self {
        Box(t)
    }
}

impl<T: MustUse> MustUse for Box<T> {
    const REASON: &'static str = T::REASON;
}

struct St;

impl MustUse for St {
    const REASON: &'static str = "because I said so";
}

fn main() {
    Box::new(St);
    //~^ ERROR unused `Box<St>` that must be used
    //~| NOTE because I said so

    Box::new(());
}
