// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C debug-assertions

#![feature(core)]

use std::slice;

fn foo<T>(v: &[T]) -> Option<&[T]> {
    let mut it = v.iter();
    for _ in 0..5 {
        let _ = it.next();
    }
    Some(it.as_slice())
}

fn foo_mut<T>(v: &mut [T]) -> Option<&mut [T]> {
    let mut it = v.iter_mut();
    for _ in 0..5 {
        let _ = it.next();
    }
    Some(it.into_slice())
}

pub fn main() {
    // In a slice of zero-size elements the pointer is meaningless.
    // Ensure iteration still works even if the pointer is at the end of the address space.
    let slice: &[()] = unsafe { slice::from_raw_parts(-5isize as *const (), 10) };
    assert_eq!(slice.len(), 10);
    assert_eq!(slice.iter().count(), 10);

    // .nth() on the iterator should also behave correctly
    let mut it = slice.iter();
    assert!(it.nth(5).is_some());
    assert_eq!(it.count(), 4);

    // Converting Iter to a slice should never have a null pointer
    assert!(foo(slice).is_some());

    // Test mutable iterators as well
    let slice: &mut [()] = unsafe { slice::from_raw_parts_mut(-5isize as *mut (), 10) };
    assert_eq!(slice.len(), 10);
    assert_eq!(slice.iter_mut().count(), 10);

    {
        let mut it = slice.iter_mut();
        assert!(it.nth(5).is_some());
        assert_eq!(it.count(), 4);
    }

    assert!(foo_mut(slice).is_some())
}
