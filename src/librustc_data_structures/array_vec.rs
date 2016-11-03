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

use std::marker::Unsize;
use std::iter::Extend;
use std::ptr::{self, drop_in_place};
use std::ops::{Deref, DerefMut, Range};
use std::hash::{Hash, Hasher};
use std::slice;
use std::fmt;
use std::mem;

pub unsafe trait Array {
    type Element;
    type PartialStorage: Default + Unsize<[ManuallyDrop<Self::Element>]>;
    const LEN: usize;
}

unsafe impl<T> Array for [T; 1] {
    type Element = T;
    type PartialStorage = [ManuallyDrop<T>; 1];
    const LEN: usize = 1;
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

impl<A> Hash for ArrayVec<A>
    where A: Array,
          A::Element: Hash {
    fn hash<H>(&self, state: &mut H) where H: Hasher {
        (&self[..]).hash(state);
    }
}

impl<A: Array> PartialEq for ArrayVec<A> {
    fn eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl<A: Array> Eq for ArrayVec<A> {}

impl<A> Clone for ArrayVec<A>
    where A: Array,
          A::Element: Clone {
    fn clone(&self) -> Self {
        let mut v = ArrayVec::new();
        v.extend(self.iter().cloned());
        v
    }
}

impl<A: Array> ArrayVec<A> {
    pub fn new() -> Self {
        ArrayVec {
            count: 0,
            values: Default::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub unsafe fn set_len(&mut self, len: usize) {
        self.count = len;
    }

    /// Panics when the stack vector is full.
    pub fn push(&mut self, el: A::Element) {
        let arr = &mut self.values as &mut [ManuallyDrop<_>];
        arr[self.count] = ManuallyDrop { value: el };
        self.count += 1;
    }

    pub fn pop(&mut self) -> Option<A::Element> {
        if self.count > 0 {
            let arr = &mut self.values as &mut [ManuallyDrop<_>];
            self.count -= 1;
            unsafe {
                let value = ptr::read(&arr[self.count]);
                Some(value.value)
            }
        } else {
            None
        }
    }
}

impl<A> Default for ArrayVec<A>
    where A: Array {
    fn default() -> Self {
        ArrayVec::new()
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
            self.push(el);
        }
    }
}

pub struct Iter<A: Array> {
    indices: Range<usize>,
    store: A::PartialStorage,
}

impl<A: Array> Drop for Iter<A> {
    fn drop(&mut self) {
        for _ in self {}
    }
}

impl<A: Array> Iterator for Iter<A> {
    type Item = A::Element;

    fn next(&mut self) -> Option<A::Element> {
        let arr = &self.store as &[ManuallyDrop<_>];
        unsafe {
            self.indices.next().map(|i| ptr::read(&arr[i]).value)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }
}

impl<A: Array> IntoIterator for ArrayVec<A> {
    type Item = A::Element;
    type IntoIter = Iter<A>;
    fn into_iter(self) -> Self::IntoIter {
        let store = unsafe {
            ptr::read(&self.values)
        };
        let indices = 0..self.count;
        mem::forget(self);
        Iter {
            indices: indices,
            store: store,
        }
    }
}

impl<'a, A: Array> IntoIterator for &'a ArrayVec<A> {
    type Item = &'a A::Element;
    type IntoIter = slice::Iter<'a, A::Element>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, A: Array> IntoIterator for &'a mut ArrayVec<A> {
    type Item = &'a mut A::Element;
    type IntoIter = slice::IterMut<'a, A::Element>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// FIXME: This should use repr(transparent) from rust-lang/rfcs#1758.
#[allow(unions_with_drop_fields)]
pub union ManuallyDrop<T> {
    value: T,
    #[allow(dead_code)]
    empty: (),
}

impl<T> ManuallyDrop<T> {
    fn new() -> ManuallyDrop<T> {
        ManuallyDrop {
            empty: ()
        }
    }
}

impl<T> Default for ManuallyDrop<T> {
    fn default() -> Self {
        ManuallyDrop::new()
    }
}

