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
use cmp::{Ord, Eq, Ordering, TotalEq, TotalOrd};
use container::{Container, Mutable};
use default::Default;
use fmt;
use iter::{DoubleEndedIterator, FromIterator, Extendable, Iterator, Rev};
use libc::{free, c_void};
use mem::{size_of, move_val_init};
use mem;
use num;
use num::{CheckedMul, CheckedAdd};
use ops::Drop;
use option::{None, Option, Some};
use ptr::RawPtr;
use ptr;
use rt::global_heap::{malloc_raw, realloc_raw};
use raw::Slice;
use vec::{ImmutableEqVector, ImmutableVector, Items, MutItems, MutableVector};
use vec::{RevItems};

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
    pub fn from_slice(values: &[T]) -> Vec<T> {
        values.iter().map(|x| x.clone()).collect()
    }

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


    pub fn grow(&mut self, n: uint, initval: &T) {
        let new_len = self.len() + n;
        self.reserve(new_len);
        let mut i: uint = 0u;

        while i < n {
            self.push((*initval).clone());
            i += 1u;
        }
    }

    pub fn grow_set(&mut self, index: uint, initval: &T, val: T) {
        let l = self.len();
        if index >= l {
            self.grow(index - l + 1u, initval);
        }
        *self.get_mut(index) = val;
    }

    pub fn partitioned(&self, f: |&T| -> bool) -> (Vec<T>, Vec<T>) {
        let mut lefts = Vec::new();
        let mut rights = Vec::new();

        for elt in self.iter() {
            if f(elt) {
                lefts.push(elt.clone());
            } else {
                rights.push(elt.clone());
            }
        }

        (lefts, rights)
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

impl<T> Extendable<T> for Vec<T> {
    fn extend<I: Iterator<T>>(&mut self, iterator: &mut I) {
        let (lower, _) = iterator.size_hint();
        self.reserve_additional(lower);
        for element in *iterator {
            self.push(element)
        }
    }
}

impl<T: Eq> Eq for Vec<T> {
    #[inline]
    fn eq(&self, other: &Vec<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Ord> Ord for Vec<T> {
    #[inline]
    fn lt(&self, other: &Vec<T>) -> bool {
        self.as_slice() < other.as_slice()
    }
}

impl<T: TotalEq> TotalEq for Vec<T> {
    #[inline]
    fn equals(&self, other: &Vec<T>) -> bool {
        self.as_slice().equals(&other.as_slice())
    }
}

impl<T: TotalOrd> TotalOrd for Vec<T> {
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

    pub fn reserve_additional(&mut self, extra: uint) {
        if self.cap - self.len < extra {
            match self.len.checked_add(&extra) {
                None => fail!("Vec::reserve_additional: `uint` overflow"),
                Some(new_cap) => self.reserve(new_cap)
            }
        }
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
    pub fn move_rev_iter(self) -> Rev<MoveItems<T>> {
        self.move_iter().rev()
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
    pub fn tailn<'a>(&'a self, n: uint) -> &'a [T] {
        self.as_slice().tailn(n)
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
    #[deprecated="Use `xs.iter().map(closure)` instead."]
    pub fn map<U>(&self, f: |t: &T| -> U) -> Vec<U> {
        self.iter().map(f).collect()
    }

    pub fn push_all_move(&mut self, other: Vec<T>) {
        for element in other.move_iter() {
            self.push(element)
        }
    }

    #[inline]
    pub fn mut_slice<'a>(&'a mut self, start: uint, end: uint)
                     -> &'a mut [T] {
        self.as_mut_slice().mut_slice(start, end)
    }

    #[inline]
    pub fn reverse(&mut self) {
        self.as_mut_slice().reverse()
    }

    #[inline]
    pub fn slice_from<'a>(&'a self, start: uint) -> &'a [T] {
        self.as_slice().slice_from(start)
    }

    #[inline]
    pub fn slice_to<'a>(&'a self, end: uint) -> &'a [T] {
        self.as_slice().slice_to(end)
    }

    #[inline]
    pub fn init<'a>(&'a self) -> &'a [T] {
        self.slice(0, self.len() - 1)
    }

    #[inline]
    pub fn as_ptr(&self) -> *T {
        self.as_slice().as_ptr()
    }
}

impl<T> Mutable for Vec<T> {
    /// Clear the vector, removing all values.
    #[inline]
    fn clear(&mut self) {
        self.truncate(0)
    }
}

impl<T:Eq> Vec<T> {
    /// Return true if a vector contains an element with the given value
    pub fn contains(&self, x: &T) -> bool {
        self.as_slice().contains(x)
    }

    pub fn dedup(&mut self) {
        unsafe {
            // Although we have a mutable reference to `self`, we cannot make
            // *arbitrary* changes. The `Eq` comparisons could fail, so we
            // must ensure that the vector is in a valid state at all time.
            //
            // The way that we handle this is by using swaps; we iterate
            // over all the elements, swapping as we go so that at the end
            // the elements we wish to keep are in the front, and those we
            // wish to reject are at the back. We can then truncate the
            // vector. This operation is still O(n).
            //
            // Example: We start in this state, where `r` represents "next
            // read" and `w` represents "next_write`.
            //
            //           r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 1 | 2 | 3 | 3 |
            //     +---+---+---+---+---+---+
            //           w
            //
            // Comparing self[r] against self[w-1], tis is not a duplicate, so
            // we swap self[r] and self[w] (no effect as r==w) and then increment both
            // r and w, leaving us with:
            //
            //               r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 1 | 2 | 3 | 3 |
            //     +---+---+---+---+---+---+
            //               w
            //
            // Comparing self[r] against self[w-1], this value is a duplicate,
            // so we increment `r` but leave everything else unchanged:
            //
            //                   r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 1 | 2 | 3 | 3 |
            //     +---+---+---+---+---+---+
            //               w
            //
            // Comparing self[r] against self[w-1], this is not a duplicate,
            // so swap self[r] and self[w] and advance r and w:
            //
            //                       r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 2 | 1 | 3 | 3 |
            //     +---+---+---+---+---+---+
            //                   w
            //
            // Not a duplicate, repeat:
            //
            //                           r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 2 | 3 | 1 | 3 |
            //     +---+---+---+---+---+---+
            //                       w
            //
            // Duplicate, advance r. End of vec. Truncate to w.

            let ln = self.len();
            if ln < 1 { return; }

            // Avoid bounds checks by using unsafe pointers.
            let p = self.as_mut_slice().as_mut_ptr();
            let mut r = 1;
            let mut w = 1;

            while r < ln {
                let p_r = p.offset(r as int);
                let p_wm1 = p.offset((w - 1) as int);
                if *p_r != *p_wm1 {
                    if r != w {
                        let p_w = p_wm1.offset(1);
                        mem::swap(&mut *p_r, &mut *p_w);
                    }
                    w += 1;
                }
                r += 1;
            }

            self.truncate(w);
        }
    }
}

#[inline]
pub fn append<T:Clone>(mut first: Vec<T>, second: &[T]) -> Vec<T> {
    first.push_all(second);
    first
}

/// Appends one element to the vector provided. The vector itself is then
/// returned for use again.
#[inline]
pub fn append_one<T>(mut lhs: Vec<T>, x: T) -> Vec<T> {
    lhs.push(x);
    lhs
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

impl<T> Default for Vec<T> {
    fn default() -> Vec<T> {
        Vec::new()
    }
}

impl<T:fmt::Show> fmt::Show for Vec<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_slice().fmt(f)
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

#[cfg(test)]
mod tests {
    use super::Vec;
    use iter::{Iterator, range, Extendable};
    use option::{Some, None};

    #[test]
    fn test_reserve_additional() {
        let mut v = Vec::new();
        assert_eq!(v.capacity(), 0);

        v.reserve_additional(2);
        assert!(v.capacity() >= 2);

        for i in range(0, 16) {
            v.push(i);
        }

        assert!(v.capacity() >= 16);
        v.reserve_additional(16);
        assert!(v.capacity() >= 32);

        v.push(16);

        v.reserve_additional(16);
        assert!(v.capacity() >= 33)
    }

    #[test]
    fn test_extend() {
        let mut v = Vec::new();
        let mut w = Vec::new();

        v.extend(&mut range(0, 3));
        for i in range(0, 3) { w.push(i) }

        assert_eq!(v, w);

        v.extend(&mut range(3, 10));
        for i in range(3, 10) { w.push(i) }

        assert_eq!(v, w);
    }
}
