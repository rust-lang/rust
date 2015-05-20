// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a very simple custom DST coercion.

#![feature(core)]

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
    let b: Bar<Baz> = a;
    unsafe {
        assert_eq!((*b.x).get(), 42);
    }
}
