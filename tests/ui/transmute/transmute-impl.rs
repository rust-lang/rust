// normalize-stderr-test "\d+ bits" -> "N bits"

// Tests that are conservative around thin/fat pointer mismatches.

#![allow(dead_code)]

use std::mem::transmute;

struct Foo<T: ?Sized> {
    t: Box<T>
}

impl<T: ?Sized> Foo<T> {
    fn m(x: &T) -> &isize where T : Sized {
        // OK here, because T : Sized is in scope.
        unsafe { transmute(x) }
    }

    fn n(x: &T) -> &isize {
        // Not OK here, because T : Sized is not in scope.
        unsafe { transmute(x) } //~ ERROR cannot transmute between types of different sizes
    }
}

fn main() { }
