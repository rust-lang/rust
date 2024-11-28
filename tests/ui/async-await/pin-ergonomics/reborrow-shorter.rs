//@check-pass

#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

use std::pin::Pin;

fn shorter<'b, T: 'b>(_: Pin<&'b mut T>) {}

fn test<'a: 'b, 'b, T: 'a>(x: Pin<&'a mut T>) {
    shorter::<'b>(x);
}

fn main() {}
