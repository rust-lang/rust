//@ check-pass

#![feature(precise_capturing_of_types)]
//~^ WARN the feature `precise_capturing_of_types` is incomplete

use std::fmt::Display;
use std::ops::Deref;

fn len<T: Deref<Target: Deref<Target = [u8]>>>(x: T) -> impl Display + use<> {
    x.len()
}

fn main() {
    let x = vec![1, 2, 3];
    let len = len(&x);
    drop(x);
    println!("len = {len}");
}
