#![allow(dead_code)]

use std::cell::Cell;

#[derive(Copy)]
struct Foo {
    x: isize
}

impl Clone for Foo {
    fn clone(&self) -> Foo {
        // Using Cell in any way should never cause clone() to be
        // invoked -- after all, that would permit evil user code to
        // abuse `Cell` and trigger crashes.

        panic!();
    }
}

pub fn main() {
    let x = Cell::new(Foo { x: 22 });
    let _y = x.get();
    let _z = x.clone();
}
