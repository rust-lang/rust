#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

// Make sure with pin reborrowing that we can only get one mutable reborrow of a pinned reference.

use std::pin::{pin, Pin};

fn twice(_: Pin<&mut i32>, _: Pin<&mut i32>) {}

fn main() {
    let x = pin!(42);
    twice(x, x); //~ ERROR cannot borrow
}
