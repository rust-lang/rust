// run-pass
#![allow(dead_code)]
use std::mem::size_of;

struct Test<T> {
    a: T
}

impl<T> Drop for Test<T> {
    fn drop(&mut self) { }
}

pub fn main() {
    assert_eq!(size_of::<isize>(), size_of::<Test<isize>>());
}
