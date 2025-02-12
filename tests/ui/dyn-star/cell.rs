// This test with Cell also indirectly exercises UnsafeCell in dyn*.
//
//@ run-pass

#![feature(dyn_star)]
#![allow(incomplete_features)]

use std::cell::Cell;

trait Rw<T> {
    fn read(&self) -> T;
    fn write(&self, v: T);
}

impl<T: Copy> Rw<T> for Cell<T> {
    fn read(&self) -> T {
        self.get()
    }
    fn write(&self, v: T) {
        self.set(v)
    }
}

fn make_dyn_star() -> dyn* Rw<usize> {
    Cell::new(42usize) as dyn* Rw<usize>
}

fn main() {
    let x = make_dyn_star();

    assert_eq!(x.read(), 42);
    x.write(24);
    assert_eq!(x.read(), 24);
}
