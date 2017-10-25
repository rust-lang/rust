// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(repr_align, attr_literals, core_intrinsics)]

use std::mem::align_of;
use std::intrinsics::type_name;

fn has_aligned_refs<'a, I, T: 'a>(iterable: I)
where
    I: Iterator<Item = &'a T>
{
    for elt in iterable {
        unsafe {
            assert_eq!((elt as *const T as usize) % align_of::<T>(), 0,
                       "Assertion failed for type {}", type_name::<T>());
        }
    }
}

fn main() {
    #[derive(Copy, Clone)]
    struct Zst;

    #[derive(Copy, Clone)]
    #[repr(align(64))]
    struct Aligned;

    has_aligned_refs([Zst; 8].iter());
    has_aligned_refs([[0f64; 0]; 8].iter());
    has_aligned_refs([Aligned; 8].iter());
    has_aligned_refs([Zst; 8].iter_mut().map(|t| &*t));
    has_aligned_refs([[0f64; 0]; 8].iter_mut().map(|t| &*t));
    has_aligned_refs([Aligned; 8].iter_mut().map(|t| &*t));
}
