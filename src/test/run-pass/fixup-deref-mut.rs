#![allow(dead_code)]
// pretty-expanded FIXME #23616

use std::ops::{Deref, DerefMut};

// Generic unique/owned smaht pointer.
struct Own<T> {
    value: *mut T
}

impl<T> Deref for Own<T> {
    type Target = T;

    fn deref<'a>(&'a self) -> &'a T {
        unsafe { &*self.value }
    }
}

impl<T> DerefMut for Own<T> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut T {
        unsafe { &mut *self.value }
    }
}

struct Point {
    x: isize,
    y: isize
}

impl Point {
    fn get(&mut self) -> (isize, isize) {
        (self.x, self.y)
    }
}

fn test0(mut x: Own<Point>) {
    let _ = x.get();
}

fn test1(mut x: Own<Own<Own<Point>>>) {
    let _ = x.get();
}

fn test2(mut x: Own<Own<Own<Point>>>) {
    let _ = (**x).get();
}

fn main() {}
