// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Migrate documentation over from `std::vec` when it is removed.
#[doc(hidden)];

use cast::{forget, transmute};
use clone::Clone;
use cmp::{Eq, Ordering, TotalEq, TotalOrd};
use container::Container;
use iter::{DoubleEndedIterator, FromIterator, Iterator};
use libc::{free, c_void};
use mem::{size_of, move_val_init};
use num;
use num::CheckedMul;
use ops::Drop;
use option::{None, Option, Some};
use ptr::RawPtr;
use ptr;
use rt::global_heap::{malloc_raw, realloc_raw};
use raw::Slice;
use vec::{ImmutableVector, Items, MutItems, MutableVector, RevItems};

pub struct Vec<T> {
    priv len: uint,
    priv cap: uint,
    priv ptr: *mut T
}

impl<T> Vec<T> {
    #[inline]
    pub fn new() -> Vec<T> {
        Vec { len: 0, cap: 0, ptr: 0 as *mut T }
    }

    pub fn with_capacity(capacity: uint) -> Vec<T> {
        if capacity == 0 {
            Vec::new()
        } else {
            let size = capacity.checked_mul(&size_of::<T>()).expect("capacity overflow");
            let ptr = unsafe { malloc_raw(size) };
            Vec { len: 0, cap: capacity, ptr: ptr as *mut T }
        }
    }

    pub fn from_fn(length: uint, op: |uint| -> T) -> Vec<T> {
        unsafe {
            let mut xs = Vec::with_capacity(length);
            while xs.len < length {
                move_val_init(xs.as_mut_slice().unsafe_mut_ref(xs.len), op(xs.len));
                xs.len += 1;
            }
            xs
        }
    }
}

impl<T: Clone> Vec<T> {
    pub fn from_elem(length: uint, value: T) -> Vec<T> {
        unsafe {
            let mut xs = Vec::with_capacity(length);
            while xs.len < length {
                move_val_init(xs.as_mut_slice().unsafe_mut_ref(xs.len), value.clone());
                xs.len += 1;
            }
            xs
        }
    }

    #[inline]
    pub fn push_all(&mut self, other: &[T]) {
        for element in other.iter() {
            self.push((*element).clone())
        }
    }
}

impl<T:Clone> Clone for Vec<T> {
    fn clone(&self) -> Vec<T> {
        let mut vector = Vec::with_capacity(self.len());
        for element in self.iter() {
            vector.push((*element).clone())
        }
        vector
    }
}

impl<T> FromIterator<T> for Vec<T> {
    fn from_iterator<I:Iterator<T>>(iterator: &mut I) -> Vec<T> {
        let (lower, _) = iterator.size_hint();
        let mut vector = Vec::with_capacity(lower);
        for element in *iterator {
            vector.push(element)
        }
        vector
    }
}

impl<T:Eq> Eq for Vec<T> {
    #[inline]
    fn eq(&self, other: &Vec<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T:TotalEq> TotalEq for Vec<T> {
    #[inline]
    fn equals(&self, other: &Vec<T>) -> bool {
        self.as_slice().equals(&other.as_slice())
    }
}

impl<T:TotalOrd> TotalOrd for Vec<T> {
    #[inline]
    fn cmp(&self, other: &Vec<T>) -> Ordering {
        self.as_slice().cmp(&other.as_slice())
    }
}

impl<T> Container for Vec<T> {
    #[inline]
    fn len(&self) -> uint {
        self.len
    }
}

impl<T> Vec<T> {
    #[inline]
    pub fn capacity(&self) -> uint {
        self.cap
    }

    pub fn reserve(&mut self, capacity: uint) {
        if capacity >= self.len {
            self.reserve_exact(num::next_power_of_two(capacity))
        }
    }

    pub fn reserve_exact(&mut self, capacity: uint) {
        if capacity >= self.len {
            let size = capacity.checked_mul(&size_of::<T>()).expect("capacity overflow");
            self.cap = capacity;
            unsafe {
                self.ptr = realloc_raw(self.ptr as *mut u8, size) as *mut T;
            }
        }
    }

    pub fn shrink_to_fit(&mut self) {
        if self.len == 0 {
            unsafe { free(self.ptr as *mut c_void) };
            self.cap = 0;
            self.ptr = 0 as *mut T;
        } else {
            unsafe {
                // Overflow check is unnecessary as the vector is already at least this large.
                self.ptr = realloc_raw(self.ptr as *mut u8, self.len * size_of::<T>()) as *mut T;
            }
            self.cap = self.len;
        }
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                self.len -= 1;
                Some(ptr::read(self.as_slice().unsafe_ref(self.len())))
            }
        }
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        if self.len == self.cap {
            if self.cap == 0 { self.cap += 2 }
            let old_size = self.cap * size_of::<T>();
            self.cap = self.cap * 2;
            let size = old_size * 2;
            if old_size > size { fail!("capacity overflow") }
            unsafe {
                self.ptr = realloc_raw(self.ptr as *mut u8, size) as *mut T;
            }
        }

        unsafe {
            let end = (self.ptr as *T).offset(self.len as int) as *mut T;
            move_val_init(&mut *end, value);
            self.len += 1;
        }
    }

    pub fn truncate(&mut self, len: uint) {
        unsafe {
            let mut i = len;
            // drop any extra elements
            while i < self.len {
                ptr::read(self.as_slice().unsafe_ref(i));
                i += 1;
            }
        }
        self.len = len;
    }

    #[inline]
    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        let slice = Slice { data: self.ptr as *T, len: self.len };
        unsafe { transmute(slice) }
    }

    #[inline]
    pub fn as_mut_slice<'a>(&'a mut self) -> &'a mut [T] {
        let slice = Slice { data: self.ptr as *T, len: self.len };
        unsafe { transmute(slice) }
    }

    #[inline]
    pub fn move_iter(self) -> MoveItems<T> {
        unsafe {
            let iter = transmute(self.as_slice().iter());
            let ptr = self.ptr as *mut c_void;
            forget(self);
            MoveItems { allocation: ptr, iter: iter }
        }
    }

    #[inline]
    pub unsafe fn set_len(&mut self, len: uint) {
        self.len = len;
    }

    #[inline]
    pub fn get<'a>(&'a self, index: uint) -> &'a T {
        &self.as_slice()[index]
    }

    #[inline]
    pub fn get_mut<'a>(&'a mut self, index: uint) -> &'a mut T {
        &mut self.as_mut_slice()[index]
    }

    #[inline]
    pub fn iter<'a>(&'a self) -> Items<'a,T> {
        self.as_slice().iter()
    }

    #[inline]
    pub fn mut_iter<'a>(&'a mut self) -> MutItems<'a,T> {
        self.as_mut_slice().mut_iter()
    }

    #[inline]
    pub fn sort_by(&mut self, compare: |&T, &T| -> Ordering) {
        self.as_mut_slice().sort_by(compare)
    }

    #[inline]
    pub fn slice<'a>(&'a self, start: uint, end: uint) -> &'a [T] {
        self.as_slice().slice(start, end)
    }

    #[inline]
    pub fn tail<'a>(&'a self) -> &'a [T] {
        self.as_slice().tail()
    }

    #[inline]
    pub fn last<'a>(&'a self) -> Option<&'a T> {
        self.as_slice().last()
    }

    #[inline]
    pub fn mut_last<'a>(&'a mut self) -> Option<&'a mut T> {
        self.as_mut_slice().mut_last()
    }

    #[inline]
    pub fn swap_remove(&mut self, index: uint) -> Option<T> {
        let length = self.len();
        if index < length - 1 {
            self.as_mut_slice().swap(index, length - 1);
        } else if index >= length {
            return None
        }
        self.pop()
    }

    #[inline]
    pub fn unshift(&mut self, element: T) {
        self.insert(0, element)
    }

    pub fn insert(&mut self, index: uint, element: T) {
        let len = self.len();
        assert!(index <= len);
        // space for the new element
        self.reserve(len + 1);

        unsafe { // infallible
            // The spot to put the new value
            {
                let slice = self.as_mut_slice();
                let p = slice.as_mut_ptr().offset(index as int);
                // Shift everything over to make space. (Duplicating the
                // `index`th element into two consecutive places.)
                ptr::copy_memory(p.offset(1), &*p, len - index);
                // Write it in, overwriting the first copy of the `index`th
                // element.
                move_val_init(&mut *p, element);
            }
            self.set_len(len + 1);
        }
    }

    #[inline]
    pub fn rev_iter<'a>(&'a self) -> RevItems<'a,T> {
        self.as_slice().rev_iter()
    }

    #[inline]
    pub fn map<U>(&self, f: |t: &T| -> U) -> Vec<U> {
        self.iter().map(f).collect()
    }

    pub fn push_all_move(&mut self, other: Vec<T>) {
        for element in other.move_iter() {
            self.push(element)
        }
    }

    pub fn slice_from<'a>(&'a self, start: uint) -> &'a [T] {
        self.as_slice().slice_from(start)
    }
}

#[inline]
pub fn append<T:Clone>(mut first: Vec<T>, second: &[T]) -> Vec<T> {
    first.push_all(second);
    first
}

#[unsafe_destructor]
impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        unsafe {
            for x in self.as_mut_slice().iter() {
                ptr::read(x);
            }
            free(self.ptr as *mut c_void)
        }
    }
}

pub struct MoveItems<T> {
    priv allocation: *mut c_void, // the block of memory allocated for the vector
    priv iter: Items<'static, T>
}

impl<T> Iterator<T> for MoveItems<T> {
    #[inline]
    fn next(&mut self) -> Option<T> {
        unsafe {
            self.iter.next().map(|x| ptr::read(x))
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

impl<T> DoubleEndedIterator<T> for MoveItems<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        unsafe {
            self.iter.next_back().map(|x| ptr::read(x))
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for MoveItems<T> {
    fn drop(&mut self) {
        // destroy the remaining elements
        for _x in *self {}
        unsafe {
            free(self.allocation)
        }
    }
}
