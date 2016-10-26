// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A stack-allocated vector, allowing storage of N elements on the stack.
//!
//! Currently, only the N = 8 case is supported (due to Array only being impl-ed for [T; 8]).

use std::marker::Unsize;
use std::iter::Extend;
use std::ptr::drop_in_place;
use std::ops::{Deref, DerefMut};
use std::slice;
use std::fmt;

pub unsafe trait Array {
    type Element;
    type PartialStorage: Default + Unsize<[ManuallyDrop<Self::Element>]>;
    const LEN: usize;
}

unsafe impl<T> Array for [T; 8] {
    type Element = T;
    type PartialStorage = [ManuallyDrop<T>; 8];
    const LEN: usize = 8;
}

pub struct ArrayVec<A: Array> {
    count: usize,
    values: A::PartialStorage
}

impl<A: Array> ArrayVec<A> {
    pub fn new() -> Self {
        ArrayVec {
            count: 0,
            values: Default::default(),
        }
    }
}

impl<A> fmt::Debug for ArrayVec<A>
    where A: Array,
          A::Element: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self[..].fmt(f)
    }
}

impl<A: Array> Deref for ArrayVec<A> {
    type Target = [A::Element];
    fn deref(&self) -> &Self::Target {
        unsafe {
            slice::from_raw_parts(&self.values as *const _ as *const A::Element, self.count)
        }
    }
}

impl<A: Array> DerefMut for ArrayVec<A> {
    fn deref_mut(&mut self) -> &mut [A::Element] {
        unsafe {
            slice::from_raw_parts_mut(&mut self.values as *mut _ as *mut A::Element, self.count)
        }
    }
}

impl<A: Array> Drop for ArrayVec<A> {
    fn drop(&mut self) {
        unsafe {
            drop_in_place(&mut self[..])
        }
    }
}

impl<A: Array> Extend<A::Element> for ArrayVec<A> {
    fn extend<I>(&mut self, iter: I) where I: IntoIterator<Item=A::Element> {
        for el in iter {
            unsafe {
                let arr = &mut self.values as &mut [ManuallyDrop<_>];
                arr[self.count].value = el;
            }
            self.count += 1;
        }
    }
}

// FIXME: This should use repr(transparent) from rust-lang/rfcs#1758.
#[allow(unions_with_drop_fields)]
pub union ManuallyDrop<T> {
    value: T,
    #[allow(dead_code)]
    empty: (),
}

impl<T> Default for ManuallyDrop<T> {
    fn default() -> Self {
        ManuallyDrop { empty: () }
    }
}

