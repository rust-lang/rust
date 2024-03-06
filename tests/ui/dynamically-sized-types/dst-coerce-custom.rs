//@ run-pass
// Test a very simple custom DST coercion.

#![feature(unsize, coerce_unsized)]

use std::ops::CoerceUnsized;
use std::marker::Unsize;

struct Bar<T: ?Sized> {
    x: *const T,
}

impl<T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<Bar<U>> for Bar<T> {}

trait Baz {
    fn get(&self) -> i32;
}

impl Baz for i32 {
    fn get(&self) -> i32 {
        *self
    }
}

fn main() {
    // Arrays.
    let a: Bar<[i32; 3]> = Bar { x: &[1, 2, 3] };
    // This is the actual coercion.
    let b: Bar<[i32]> = a;

    unsafe {
        assert_eq!((*b.x)[0], 1);
        assert_eq!((*b.x)[1], 2);
        assert_eq!((*b.x)[2], 3);
    }

    // Trait objects.
    let a: Bar<i32> = Bar { x: &42 };
    let b: Bar<dyn Baz> = a;
    unsafe {
        assert_eq!((*b.x).get(), 42);
    }
}
