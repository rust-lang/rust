// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! VecDeque is a double-ended queue, which is implemented with the help of a
//! growing ring buffer.
//!
//! This queue has `O(1)` amortized inserts and removals from both ends of the
//! container. It also has `O(1)` indexing like a vector. The contained elements
//! are not required to be copyable, and the queue will be sendable if the
//! contained type is sendable.

#![stable(feature = "rust1", since = "1.0.0")]

use core::cmp::Ordering;
use core::fmt;
use core::iter::{repeat, FromIterator, FusedIterator};
use core::mem;
use core::ops::{Index, IndexMut};
use core::ptr;
use core::ptr::Shared;
use core::slice;

use core::hash::{Hash, Hasher};
use core::cmp;

use alloc::raw_vec::RawVec;

use super::range::RangeArgument;
use Bound::{Excluded, Included, Unbounded};
use super::vec::Vec;

const INITIAL_CAPACITY: usize = 7; // 2^3 - 1
const MINIMUM_CAPACITY: usize = 1; // 2 - 1
#[cfg(target_pointer_width = "32")]
const MAXIMUM_ZST_CAPACITY: usize = 1 << (32 - 1); // Largest possible power of two
#[cfg(target_pointer_width = "64")]
const MAXIMUM_ZST_CAPACITY: usize = 1 << (64 - 1); // Largest possible power of two

/// `VecDeque` is a growable ring buffer, which can be used as a double-ended
/// queue efficiently.
///
/// The "default" usage of this type as a queue is to use `push_back` to add to
/// the queue, and `pop_front` to remove from the queue. `extend` and `append`
/// push onto the back in this manner, and iterating over `VecDeque` goes front
/// to back.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct VecDeque<T> {
    // tail and head are pointers into the buffer. Tail always points
    // to the first element that could be read, Head always points
    // to where data should be written.
    // If tail == head the buffer is empty. The length of the ringbuffer
    // is defined as the distance between the two.
    tail: usize,
    head: usize,
    buf: RawVec<T>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone> Clone for VecDeque<T> {
    fn clone(&self) -> VecDeque<T> {
        self.iter().cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<#[may_dangle] T> Drop for VecDeque<T> {
    fn drop(&mut self) {
        let (front, back) = self.as_mut_slices();
        unsafe {
            // use drop for [T]
            ptr::drop_in_place(front);
            ptr::drop_in_place(back);
        }
        // RawVec handles deallocation
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for VecDeque<T> {
    /// Creates an empty `VecDeque<T>`.
    #[inline]
    fn default() -> VecDeque<T> {
        VecDeque::new()
    }
}

impl<T> VecDeque<T> {
    /// Marginally more convenient
    #[inline]
    fn ptr(&self) -> *mut T {
        self.buf.ptr()
    }

    /// Marginally more convenient
    #[inline]
    fn cap(&self) -> usize {
        if mem::size_of::<T>() == 0 {
            // For zero sized types, we are always at maximum capacity
            MAXIMUM_ZST_CAPACITY
        } else {
            self.buf.cap()
        }
    }

    /// Turn ptr into a slice
    #[inline]
    unsafe fn buffer_as_slice(&self) -> &[T] {
        slice::from_raw_parts(self.ptr(), self.cap())
    }

    /// Turn ptr into a mut slice
    #[inline]
    unsafe fn buffer_as_mut_slice(&mut self) -> &mut [T] {
        slice::from_raw_parts_mut(self.ptr(), self.cap())
    }

    /// Moves an element out of the buffer
    #[inline]
    unsafe fn buffer_read(&mut self, off: usize) -> T {
        ptr::read(self.ptr().offset(off as isize))
    }

    /// Writes an element into the buffer, moving it.
    #[inline]
    unsafe fn buffer_write(&mut self, off: usize, value: T) {
        ptr::write(self.ptr().offset(off as isize), value);
    }

    /// Returns true if and only if the buffer is at capacity
    #[inline]
    fn is_full(&self) -> bool {
        self.cap() - self.len() == 1
    }

    /// Returns the index in the underlying buffer for a given logical element
    /// index.
    #[inline]
    fn wrap_index(&self, idx: usize) -> usize {
        wrap_index(idx, self.cap())
    }

    /// Returns the index in the underlying buffer for a given logical element
    /// index + addend.
    #[inline]
    fn wrap_add(&self, idx: usize, addend: usize) -> usize {
        wrap_index(idx.wrapping_add(addend), self.cap())
    }

    /// Returns the index in the underlying buffer for a given logical element
    /// index - subtrahend.
    #[inline]
    fn wrap_sub(&self, idx: usize, subtrahend: usize) -> usize {
        wrap_index(idx.wrapping_sub(subtrahend), self.cap())
    }

    /// Copies a contiguous block of memory len long from src to dst
    #[inline]
    unsafe fn copy(&self, dst: usize, src: usize, len: usize) {
        debug_assert!(dst + len <= self.cap(),
                      "cpy dst={} src={} len={} cap={}",
                      dst,
                      src,
                      len,
                      self.cap());
        debug_assert!(src + len <= self.cap(),
                      "cpy dst={} src={} len={} cap={}",
                      dst,
                      src,
                      len,
                      self.cap());
        ptr::copy(self.ptr().offset(src as isize),
                  self.ptr().offset(dst as isize),
                  len);
    }

    /// Copies a contiguous block of memory len long from src to dst
    #[inline]
    unsafe fn copy_nonoverlapping(&self, dst: usize, src: usize, len: usize) {
        debug_assert!(dst + len <= self.cap(),
                      "cno dst={} src={} len={} cap={}",
                      dst,
                      src,
                      len,
                      self.cap());
        debug_assert!(src + len <= self.cap(),
                      "cno dst={} src={} len={} cap={}",
                      dst,
                      src,
                      len,
                      self.cap());
        ptr::copy_nonoverlapping(self.ptr().offset(src as isize),
                                 self.ptr().offset(dst as isize),
                                 len);
    }

    /// Copies a potentially wrapping block of memory len long from src to dest.
    /// (abs(dst - src) + len) must be no larger than cap() (There must be at
    /// most one continuous overlapping region between src and dest).
    unsafe fn wrap_copy(&self, dst: usize, src: usize, len: usize) {
        #[allow(dead_code)]
        fn diff(a: usize, b: usize) -> usize {
            if a <= b { b - a } else { a - b }
        }
        debug_assert!(cmp::min(diff(dst, src), self.cap() - diff(dst, src)) + len <= self.cap(),
                      "wrc dst={} src={} len={} cap={}",
                      dst,
                      src,
                      len,
                      self.cap());

        if src == dst || len == 0 {
            return;
        }

        let dst_after_src = self.wrap_sub(dst, src) < len;

        let src_pre_wrap_len = self.cap() - src;
        let dst_pre_wrap_len = self.cap() - dst;
        let src_wraps = src_pre_wrap_len < len;
        let dst_wraps = dst_pre_wrap_len < len;

        match (dst_after_src, src_wraps, dst_wraps) {
            (_, false, false) => {
                // src doesn't wrap, dst doesn't wrap
                //
                //        S . . .
                // 1 [_ _ A A B B C C _]
                // 2 [_ _ A A A A B B _]
                //            D . . .
                //
                self.copy(dst, src, len);
            }
            (false, false, true) => {
                // dst before src, src doesn't wrap, dst wraps
                //
                //    S . . .
                // 1 [A A B B _ _ _ C C]
                // 2 [A A B B _ _ _ A A]
                // 3 [B B B B _ _ _ A A]
                //    . .           D .
                //
                self.copy(dst, src, dst_pre_wrap_len);
                self.copy(0, src + dst_pre_wrap_len, len - dst_pre_wrap_len);
            }
            (true, false, true) => {
                // src before dst, src doesn't wrap, dst wraps
                //
                //              S . . .
                // 1 [C C _ _ _ A A B B]
                // 2 [B B _ _ _ A A B B]
                // 3 [B B _ _ _ A A A A]
                //    . .           D .
                //
                self.copy(0, src + dst_pre_wrap_len, len - dst_pre_wrap_len);
                self.copy(dst, src, dst_pre_wrap_len);
            }
            (false, true, false) => {
                // dst before src, src wraps, dst doesn't wrap
                //
                //    . .           S .
                // 1 [C C _ _ _ A A B B]
                // 2 [C C _ _ _ B B B B]
                // 3 [C C _ _ _ B B C C]
                //              D . . .
                //
                self.copy(dst, src, src_pre_wrap_len);
                self.copy(dst + src_pre_wrap_len, 0, len - src_pre_wrap_len);
            }
            (true, true, false) => {
                // src before dst, src wraps, dst doesn't wrap
                //
                //    . .           S .
                // 1 [A A B B _ _ _ C C]
                // 2 [A A A A _ _ _ C C]
                // 3 [C C A A _ _ _ C C]
                //    D . . .
                //
                self.copy(dst + src_pre_wrap_len, 0, len - src_pre_wrap_len);
                self.copy(dst, src, src_pre_wrap_len);
            }
            (false, true, true) => {
                // dst before src, src wraps, dst wraps
                //
                //    . . .         S .
                // 1 [A B C D _ E F G H]
                // 2 [A B C D _ E G H H]
                // 3 [A B C D _ E G H A]
                // 4 [B C C D _ E G H A]
                //    . .         D . .
                //
                debug_assert!(dst_pre_wrap_len > src_pre_wrap_len);
                let delta = dst_pre_wrap_len - src_pre_wrap_len;
                self.copy(dst, src, src_pre_wrap_len);
                self.copy(dst + src_pre_wrap_len, 0, delta);
                self.copy(0, delta, len - dst_pre_wrap_len);
            }
            (true, true, true) => {
                // src before dst, src wraps, dst wraps
                //
                //    . .         S . .
                // 1 [A B C D _ E F G H]
                // 2 [A A B D _ E F G H]
                // 3 [H A B D _ E F G H]
                // 4 [H A B D _ E F F G]
                //    . . .         D .
                //
                debug_assert!(src_pre_wrap_len > dst_pre_wrap_len);
                let delta = src_pre_wrap_len - dst_pre_wrap_len;
                self.copy(delta, 0, len - src_pre_wrap_len);
                self.copy(0, self.cap() - delta, delta);
                self.copy(dst, src, dst_pre_wrap_len);
            }
        }
    }

    /// Frobs the head and tail sections around to handle the fact that we
    /// just reallocated. Unsafe because it trusts old_cap.
    #[inline]
    unsafe fn handle_cap_increase(&mut self, old_cap: usize) {
        let new_cap = self.cap();

        // Move the shortest contiguous section of the ring buffer
        //    T             H
        //   [o o o o o o o . ]
        //    T             H
        // A [o o o o o o o . . . . . . . . . ]
        //        H T
        //   [o o . o o o o o ]
        //          T             H
        // B [. . . o o o o o o o . . . . . . ]
        //              H T
        //   [o o o o o . o o ]
        //              H                 T
        // C [o o o o o . . . . . . . . . o o ]

        if self.tail <= self.head {
            // A
            // Nop
        } else if self.head < old_cap - self.tail {
            // B
            self.copy_nonoverlapping(old_cap, 0, self.head);
            self.head += old_cap;
            debug_assert!(self.head > self.tail);
        } else {
            // C
            let new_tail = new_cap - (old_cap - self.tail);
            self.copy_nonoverlapping(new_tail, self.tail, old_cap - self.tail);
            self.tail = new_tail;
            debug_assert!(self.head < self.tail);
        }
        debug_assert!(self.head < self.cap());
        debug_assert!(self.tail < self.cap());
        debug_assert!(self.cap().count_ones() == 1);
    }
}

impl<T> VecDeque<T> {
    /// Creates an empty `VecDeque`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let vector: VecDeque<u32> = VecDeque::new();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> VecDeque<T> {
        VecDeque::with_capacity(INITIAL_CAPACITY)
    }

    /// Creates an empty `VecDeque` with space for at least `n` elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let vector: VecDeque<u32> = VecDeque::with_capacity(10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(n: usize) -> VecDeque<T> {
        // +1 since the ringbuffer always leaves one space empty
        let cap = cmp::max(n + 1, MINIMUM_CAPACITY + 1).next_power_of_two();
        assert!(cap > n, "capacity overflow");

        VecDeque {
            tail: 0,
            head: 0,
            buf: RawVec::with_capacity(cap),
        }
    }

    /// Retrieves an element in the `VecDeque` by index.
    ///
    /// Element at index 0 is the front of the queue.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf = VecDeque::new();
    /// buf.push_back(3);
    /// buf.push_back(4);
    /// buf.push_back(5);
    /// assert_eq!(buf.get(1), Some(&4));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len() {
            let idx = self.wrap_add(self.tail, index);
            unsafe { Some(&*self.ptr().offset(idx as isize)) }
        } else {
            None
        }
    }

    /// Retrieves an element in the `VecDeque` mutably by index.
    ///
    /// Element at index 0 is the front of the queue.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf = VecDeque::new();
    /// buf.push_back(3);
    /// buf.push_back(4);
    /// buf.push_back(5);
    /// if let Some(elem) = buf.get_mut(1) {
    ///     *elem = 7;
    /// }
    ///
    /// assert_eq!(buf[1], 7);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len() {
            let idx = self.wrap_add(self.tail, index);
            unsafe { Some(&mut *self.ptr().offset(idx as isize)) }
        } else {
            None
        }
    }

    /// Swaps elements at indices `i` and `j`.
    ///
    /// `i` and `j` may be equal.
    ///
    /// Fails if there is no element with either index.
    ///
    /// Element at index 0 is the front of the queue.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf = VecDeque::new();
    /// buf.push_back(3);
    /// buf.push_back(4);
    /// buf.push_back(5);
    /// buf.swap(0, 2);
    /// assert_eq!(buf[0], 5);
    /// assert_eq!(buf[2], 3);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn swap(&mut self, i: usize, j: usize) {
        assert!(i < self.len());
        assert!(j < self.len());
        let ri = self.wrap_add(self.tail, i);
        let rj = self.wrap_add(self.tail, j);
        unsafe {
            ptr::swap(self.ptr().offset(ri as isize),
                      self.ptr().offset(rj as isize))
        }
    }

    /// Returns the number of elements the `VecDeque` can hold without
    /// reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let buf: VecDeque<i32> = VecDeque::with_capacity(10);
    /// assert!(buf.capacity() >= 10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn capacity(&self) -> usize {
        self.cap() - 1
    }

    /// Reserves the minimum capacity for exactly `additional` more elements to be inserted in the
    /// given `VecDeque`. Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore
    /// capacity can not be relied upon to be precisely minimal. Prefer `reserve` if future
    /// insertions are expected.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf: VecDeque<i32> = vec![1].into_iter().collect();
    /// buf.reserve_exact(10);
    /// assert!(buf.capacity() >= 11);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.reserve(additional);
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in the given
    /// `VecDeque`. The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf: VecDeque<i32> = vec![1].into_iter().collect();
    /// buf.reserve(10);
    /// assert!(buf.capacity() >= 11);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve(&mut self, additional: usize) {
        let old_cap = self.cap();
        let used_cap = self.len() + 1;
        let new_cap = used_cap.checked_add(additional)
            .and_then(|needed_cap| needed_cap.checked_next_power_of_two())
            .expect("capacity overflow");

        if new_cap > self.capacity() {
            self.buf.reserve_exact(used_cap, new_cap - used_cap);
            unsafe {
                self.handle_cap_increase(old_cap);
            }
        }
    }

    /// Shrinks the capacity of the `VecDeque` as much as possible.
    ///
    /// It will drop down as close as possible to the length but the allocator may still inform the
    /// `VecDeque` that there is space for a few more elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf = VecDeque::with_capacity(15);
    /// buf.extend(0..4);
    /// assert_eq!(buf.capacity(), 15);
    /// buf.shrink_to_fit();
    /// assert!(buf.capacity() >= 4);
    /// ```
    #[stable(feature = "deque_extras_15", since = "1.5.0")]
    pub fn shrink_to_fit(&mut self) {
        // +1 since the ringbuffer always leaves one space empty
        // len + 1 can't overflow for an existing, well-formed ringbuffer.
        let target_cap = cmp::max(self.len() + 1, MINIMUM_CAPACITY + 1).next_power_of_two();
        if target_cap < self.cap() {
            // There are three cases of interest:
            //   All elements are out of desired bounds
            //   Elements are contiguous, and head is out of desired bounds
            //   Elements are discontiguous, and tail is out of desired bounds
            //
            // At all other times, element positions are unaffected.
            //
            // Indicates that elements at the head should be moved.
            let head_outside = self.head == 0 || self.head >= target_cap;
            // Move elements from out of desired bounds (positions after target_cap)
            if self.tail >= target_cap && head_outside {
                //                    T             H
                //   [. . . . . . . . o o o o o o o . ]
                //    T             H
                //   [o o o o o o o . ]
                unsafe {
                    self.copy_nonoverlapping(0, self.tail, self.len());
                }
                self.head = self.len();
                self.tail = 0;
            } else if self.tail != 0 && self.tail < target_cap && head_outside {
                //          T             H
                //   [. . . o o o o o o o . . . . . . ]
                //        H T
                //   [o o . o o o o o ]
                let len = self.wrap_sub(self.head, target_cap);
                unsafe {
                    self.copy_nonoverlapping(0, target_cap, len);
                }
                self.head = len;
                debug_assert!(self.head < self.tail);
            } else if self.tail >= target_cap {
                //              H                 T
                //   [o o o o o . . . . . . . . . o o ]
                //              H T
                //   [o o o o o . o o ]
                debug_assert!(self.wrap_sub(self.head, 1) < target_cap);
                let len = self.cap() - self.tail;
                let new_tail = target_cap - len;
                unsafe {
                    self.copy_nonoverlapping(new_tail, self.tail, len);
                }
                self.tail = new_tail;
                debug_assert!(self.head < self.tail);
            }

            self.buf.shrink_to_fit(target_cap);

            debug_assert!(self.head < self.cap());
            debug_assert!(self.tail < self.cap());
            debug_assert!(self.cap().count_ones() == 1);
        }
    }

    /// Shortens a `VecDeque`, dropping excess elements from the back.
    ///
    /// If `len` is greater than the `VecDeque`'s current length, this has no
    /// effect.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(deque_extras)]
    ///
    /// use std::collections::VecDeque;
    ///
    /// let mut buf = VecDeque::new();
    /// buf.push_back(5);
    /// buf.push_back(10);
    /// buf.push_back(15);
    /// buf.truncate(1);
    /// assert_eq!(buf.len(), 1);
    /// assert_eq!(Some(&5), buf.get(0));
    /// ```
    #[unstable(feature = "deque_extras",
               reason = "matches collection reform specification; waiting on panic semantics",
               issue = "27788")]
    pub fn truncate(&mut self, len: usize) {
        for _ in len..self.len() {
            self.pop_back();
        }
    }

    /// Returns a front-to-back iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf = VecDeque::new();
    /// buf.push_back(5);
    /// buf.push_back(3);
    /// buf.push_back(4);
    /// let b: &[_] = &[&5, &3, &4];
    /// let c: Vec<&i32> = buf.iter().collect();
    /// assert_eq!(&c[..], b);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<T> {
        Iter {
            tail: self.tail,
            head: self.head,
            ring: unsafe { self.buffer_as_slice() },
        }
    }

    /// Returns a front-to-back iterator that returns mutable references.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf = VecDeque::new();
    /// buf.push_back(5);
    /// buf.push_back(3);
    /// buf.push_back(4);
    /// for num in buf.iter_mut() {
    ///     *num = *num - 2;
    /// }
    /// let b: &[_] = &[&mut 3, &mut 1, &mut 2];
    /// assert_eq!(&buf.iter_mut().collect::<Vec<&mut i32>>()[..], b);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            tail: self.tail,
            head: self.head,
            ring: unsafe { self.buffer_as_mut_slice() },
        }
    }

    /// Returns a pair of slices which contain, in order, the contents of the
    /// `VecDeque`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut vector = VecDeque::new();
    ///
    /// vector.push_back(0);
    /// vector.push_back(1);
    /// vector.push_back(2);
    ///
    /// assert_eq!(vector.as_slices(), (&[0, 1, 2][..], &[][..]));
    ///
    /// vector.push_front(10);
    /// vector.push_front(9);
    ///
    /// assert_eq!(vector.as_slices(), (&[9, 10][..], &[0, 1, 2][..]));
    /// ```
    #[inline]
    #[stable(feature = "deque_extras_15", since = "1.5.0")]
    pub fn as_slices(&self) -> (&[T], &[T]) {
        unsafe {
            let buf = self.buffer_as_slice();
            RingSlices::ring_slices(buf, self.head, self.tail)
        }
    }

    /// Returns a pair of slices which contain, in order, the contents of the
    /// `VecDeque`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut vector = VecDeque::new();
    ///
    /// vector.push_back(0);
    /// vector.push_back(1);
    ///
    /// vector.push_front(10);
    /// vector.push_front(9);
    ///
    /// vector.as_mut_slices().0[0] = 42;
    /// vector.as_mut_slices().1[0] = 24;
    /// assert_eq!(vector.as_slices(), (&[42, 10][..], &[24, 1][..]));
    /// ```
    #[inline]
    #[stable(feature = "deque_extras_15", since = "1.5.0")]
    pub fn as_mut_slices(&mut self) -> (&mut [T], &mut [T]) {
        unsafe {
            let head = self.head;
            let tail = self.tail;
            let buf = self.buffer_as_mut_slice();
            RingSlices::ring_slices(buf, head, tail)
        }
    }

    /// Returns the number of elements in the `VecDeque`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut v = VecDeque::new();
    /// assert_eq!(v.len(), 0);
    /// v.push_back(1);
    /// assert_eq!(v.len(), 1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize {
        count(self.tail, self.head, self.cap())
    }

    /// Returns true if the buffer contains no elements
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut v = VecDeque::new();
    /// assert!(v.is_empty());
    /// v.push_front(1);
    /// assert!(!v.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool {
        self.tail == self.head
    }

    /// Create a draining iterator that removes the specified range in the
    /// `VecDeque` and yields the removed items.
    ///
    /// Note 1: The element range is removed even if the iterator is not
    /// consumed until the end.
    ///
    /// Note 2: It is unspecified how many elements are removed from the deque,
    /// if the `Drain` value is not dropped, but the borrow it holds expires
    /// (eg. due to mem::forget).
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut v: VecDeque<_> = vec![1, 2, 3].into_iter().collect();
    /// assert_eq!(vec![3].into_iter().collect::<VecDeque<_>>(), v.drain(2..).collect());
    /// assert_eq!(vec![1, 2].into_iter().collect::<VecDeque<_>>(), v);
    ///
    /// // A full range clears all contents
    /// v.drain(..);
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    #[stable(feature = "drain", since = "1.6.0")]
    pub fn drain<R>(&mut self, range: R) -> Drain<T>
        where R: RangeArgument<usize>
    {
        // Memory safety
        //
        // When the Drain is first created, the source deque is shortened to
        // make sure no uninitialized or moved-from elements are accessible at
        // all if the Drain's destructor never gets to run.
        //
        // Drain will ptr::read out the values to remove.
        // When finished, the remaining data will be copied back to cover the hole,
        // and the head/tail values will be restored correctly.
        //
        let len = self.len();
        let start = match range.start() {
            Included(&n) => n,
            Excluded(&n) => n + 1,
            Unbounded    => 0,
        };
        let end = match range.end() {
            Included(&n) => n + 1,
            Excluded(&n) => n,
            Unbounded    => len,
        };
        assert!(start <= end, "drain lower bound was too large");
        assert!(end <= len, "drain upper bound was too large");

        // The deque's elements are parted into three segments:
        // * self.tail  -> drain_tail
        // * drain_tail -> drain_head
        // * drain_head -> self.head
        //
        // T = self.tail; H = self.head; t = drain_tail; h = drain_head
        //
        // We store drain_tail as self.head, and drain_head and self.head as
        // after_tail and after_head respectively on the Drain. This also
        // truncates the effective array such that if the Drain is leaked, we
        // have forgotten about the potentially moved values after the start of
        // the drain.
        //
        //        T   t   h   H
        // [. . . o o x x o o . . .]
        //
        let drain_tail = self.wrap_add(self.tail, start);
        let drain_head = self.wrap_add(self.tail, end);
        let head = self.head;

        // "forget" about the values after the start of the drain until after
        // the drain is complete and the Drain destructor is run.
        self.head = drain_tail;

        Drain {
            deque: unsafe { Shared::new(self as *mut _) },
            after_tail: drain_head,
            after_head: head,
            iter: Iter {
                tail: drain_tail,
                head: drain_head,
                ring: unsafe { self.buffer_as_mut_slice() },
            },
        }
    }

    /// Clears the buffer, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut v = VecDeque::new();
    /// v.push_back(1);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn clear(&mut self) {
        self.drain(..);
    }

    /// Returns `true` if the `VecDeque` contains an element equal to the
    /// given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut vector: VecDeque<u32> = VecDeque::new();
    ///
    /// vector.push_back(0);
    /// vector.push_back(1);
    ///
    /// assert_eq!(vector.contains(&1), true);
    /// assert_eq!(vector.contains(&10), false);
    /// ```
    #[stable(feature = "vec_deque_contains", since = "1.12.0")]
    pub fn contains(&self, x: &T) -> bool
        where T: PartialEq<T>
    {
        let (a, b) = self.as_slices();
        a.contains(x) || b.contains(x)
    }

    /// Provides a reference to the front element, or `None` if the sequence is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut d = VecDeque::new();
    /// assert_eq!(d.front(), None);
    ///
    /// d.push_back(1);
    /// d.push_back(2);
    /// assert_eq!(d.front(), Some(&1));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn front(&self) -> Option<&T> {
        if !self.is_empty() {
            Some(&self[0])
        } else {
            None
        }
    }

    /// Provides a mutable reference to the front element, or `None` if the
    /// sequence is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut d = VecDeque::new();
    /// assert_eq!(d.front_mut(), None);
    ///
    /// d.push_back(1);
    /// d.push_back(2);
    /// match d.front_mut() {
    ///     Some(x) => *x = 9,
    ///     None => (),
    /// }
    /// assert_eq!(d.front(), Some(&9));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn front_mut(&mut self) -> Option<&mut T> {
        if !self.is_empty() {
            Some(&mut self[0])
        } else {
            None
        }
    }

    /// Provides a reference to the back element, or `None` if the sequence is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut d = VecDeque::new();
    /// assert_eq!(d.back(), None);
    ///
    /// d.push_back(1);
    /// d.push_back(2);
    /// assert_eq!(d.back(), Some(&2));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn back(&self) -> Option<&T> {
        if !self.is_empty() {
            Some(&self[self.len() - 1])
        } else {
            None
        }
    }

    /// Provides a mutable reference to the back element, or `None` if the
    /// sequence is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut d = VecDeque::new();
    /// assert_eq!(d.back(), None);
    ///
    /// d.push_back(1);
    /// d.push_back(2);
    /// match d.back_mut() {
    ///     Some(x) => *x = 9,
    ///     None => (),
    /// }
    /// assert_eq!(d.back(), Some(&9));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn back_mut(&mut self) -> Option<&mut T> {
        let len = self.len();
        if !self.is_empty() {
            Some(&mut self[len - 1])
        } else {
            None
        }
    }

    /// Removes the first element and returns it, or `None` if the sequence is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut d = VecDeque::new();
    /// d.push_back(1);
    /// d.push_back(2);
    ///
    /// assert_eq!(d.pop_front(), Some(1));
    /// assert_eq!(d.pop_front(), Some(2));
    /// assert_eq!(d.pop_front(), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let tail = self.tail;
            self.tail = self.wrap_add(self.tail, 1);
            unsafe { Some(self.buffer_read(tail)) }
        }
    }

    /// Inserts an element first in the sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut d = VecDeque::new();
    /// d.push_front(1);
    /// d.push_front(2);
    /// assert_eq!(d.front(), Some(&2));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push_front(&mut self, value: T) {
        if self.is_full() {
            let old_cap = self.cap();
            self.buf.double();
            unsafe {
                self.handle_cap_increase(old_cap);
            }
            debug_assert!(!self.is_full());
        }

        self.tail = self.wrap_sub(self.tail, 1);
        let tail = self.tail;
        unsafe {
            self.buffer_write(tail, value);
        }
    }

    /// Appends an element to the back of a buffer
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf = VecDeque::new();
    /// buf.push_back(1);
    /// buf.push_back(3);
    /// assert_eq!(3, *buf.back().unwrap());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push_back(&mut self, value: T) {
        if self.is_full() {
            let old_cap = self.cap();
            self.buf.double();
            unsafe {
                self.handle_cap_increase(old_cap);
            }
            debug_assert!(!self.is_full());
        }

        let head = self.head;
        self.head = self.wrap_add(self.head, 1);
        unsafe { self.buffer_write(head, value) }
    }

    /// Removes the last element from a buffer and returns it, or `None` if
    /// it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf = VecDeque::new();
    /// assert_eq!(buf.pop_back(), None);
    /// buf.push_back(1);
    /// buf.push_back(3);
    /// assert_eq!(buf.pop_back(), Some(3));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            self.head = self.wrap_sub(self.head, 1);
            let head = self.head;
            unsafe { Some(self.buffer_read(head)) }
        }
    }

    #[inline]
    fn is_contiguous(&self) -> bool {
        self.tail <= self.head
    }

    /// Removes an element from anywhere in the `VecDeque` and returns it, replacing it with the
    /// last element.
    ///
    /// This does not preserve ordering, but is O(1).
    ///
    /// Returns `None` if `index` is out of bounds.
    ///
    /// Element at index 0 is the front of the queue.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf = VecDeque::new();
    /// assert_eq!(buf.swap_remove_back(0), None);
    /// buf.push_back(1);
    /// buf.push_back(2);
    /// buf.push_back(3);
    ///
    /// assert_eq!(buf.swap_remove_back(0), Some(1));
    /// assert_eq!(buf.len(), 2);
    /// assert_eq!(buf[0], 3);
    /// assert_eq!(buf[1], 2);
    /// ```
    #[stable(feature = "deque_extras_15", since = "1.5.0")]
    pub fn swap_remove_back(&mut self, index: usize) -> Option<T> {
        let length = self.len();
        if length > 0 && index < length - 1 {
            self.swap(index, length - 1);
        } else if index >= length {
            return None;
        }
        self.pop_back()
    }

    /// Removes an element from anywhere in the `VecDeque` and returns it,
    /// replacing it with the first element.
    ///
    /// This does not preserve ordering, but is O(1).
    ///
    /// Returns `None` if `index` is out of bounds.
    ///
    /// Element at index 0 is the front of the queue.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf = VecDeque::new();
    /// assert_eq!(buf.swap_remove_front(0), None);
    /// buf.push_back(1);
    /// buf.push_back(2);
    /// buf.push_back(3);
    ///
    /// assert_eq!(buf.swap_remove_front(2), Some(3));
    /// assert_eq!(buf.len(), 2);
    /// assert_eq!(buf[0], 2);
    /// assert_eq!(buf[1], 1);
    /// ```
    #[stable(feature = "deque_extras_15", since = "1.5.0")]
    pub fn swap_remove_front(&mut self, index: usize) -> Option<T> {
        let length = self.len();
        if length > 0 && index < length && index != 0 {
            self.swap(index, 0);
        } else if index >= length {
            return None;
        }
        self.pop_front()
    }

    /// Inserts an element at `index` within the `VecDeque`, shifting all elements with indices
    /// greater than or equal to `index` towards the back.
    ///
    /// Element at index 0 is the front of the queue.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than `VecDeque`'s length
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut vec_deque = VecDeque::new();
    /// vec_deque.push_back('a');
    /// vec_deque.push_back('b');
    /// vec_deque.push_back('c');
    ///
    /// vec_deque.insert(1, 'd');
    ///
    /// let vec = vec_deque.into_iter().collect::<Vec<_>>();
    /// assert_eq!(vec, ['a', 'd', 'b', 'c']);
    /// ```
    #[stable(feature = "deque_extras_15", since = "1.5.0")]
    pub fn insert(&mut self, index: usize, value: T) {
        assert!(index <= self.len(), "index out of bounds");
        if self.is_full() {
            let old_cap = self.cap();
            self.buf.double();
            unsafe {
                self.handle_cap_increase(old_cap);
            }
            debug_assert!(!self.is_full());
        }

        // Move the least number of elements in the ring buffer and insert
        // the given object
        //
        // At most len/2 - 1 elements will be moved. O(min(n, n-i))
        //
        // There are three main cases:
        //  Elements are contiguous
        //      - special case when tail is 0
        //  Elements are discontiguous and the insert is in the tail section
        //  Elements are discontiguous and the insert is in the head section
        //
        // For each of those there are two more cases:
        //  Insert is closer to tail
        //  Insert is closer to head
        //
        // Key: H - self.head
        //      T - self.tail
        //      o - Valid element
        //      I - Insertion element
        //      A - The element that should be after the insertion point
        //      M - Indicates element was moved

        let idx = self.wrap_add(self.tail, index);

        let distance_to_tail = index;
        let distance_to_head = self.len() - index;

        let contiguous = self.is_contiguous();

        match (contiguous, distance_to_tail <= distance_to_head, idx >= self.tail) {
            (true, true, _) if index == 0 => {
                // push_front
                //
                //       T
                //       I             H
                //      [A o o o o o o . . . . . . . . .]
                //
                //                       H         T
                //      [A o o o o o o o . . . . . I]
                //

                self.tail = self.wrap_sub(self.tail, 1);
            }
            (true, true, _) => {
                unsafe {
                    // contiguous, insert closer to tail:
                    //
                    //             T   I         H
                    //      [. . . o o A o o o o . . . . . .]
                    //
                    //           T               H
                    //      [. . o o I A o o o o . . . . . .]
                    //           M M
                    //
                    // contiguous, insert closer to tail and tail is 0:
                    //
                    //
                    //       T   I         H
                    //      [o o A o o o o . . . . . . . . .]
                    //
                    //                       H             T
                    //      [o I A o o o o o . . . . . . . o]
                    //       M                             M

                    let new_tail = self.wrap_sub(self.tail, 1);

                    self.copy(new_tail, self.tail, 1);
                    // Already moved the tail, so we only copy `index - 1` elements.
                    self.copy(self.tail, self.tail + 1, index - 1);

                    self.tail = new_tail;
                }
            }
            (true, false, _) => {
                unsafe {
                    //  contiguous, insert closer to head:
                    //
                    //             T       I     H
                    //      [. . . o o o o A o o . . . . . .]
                    //
                    //             T               H
                    //      [. . . o o o o I A o o . . . . .]
                    //                       M M M

                    self.copy(idx + 1, idx, self.head - idx);
                    self.head = self.wrap_add(self.head, 1);
                }
            }
            (false, true, true) => {
                unsafe {
                    // discontiguous, insert closer to tail, tail section:
                    //
                    //                   H         T   I
                    //      [o o o o o o . . . . . o o A o o]
                    //
                    //                   H       T
                    //      [o o o o o o . . . . o o I A o o]
                    //                           M M

                    self.copy(self.tail - 1, self.tail, index);
                    self.tail -= 1;
                }
            }
            (false, false, true) => {
                unsafe {
                    // discontiguous, insert closer to head, tail section:
                    //
                    //           H             T         I
                    //      [o o . . . . . . . o o o o o A o]
                    //
                    //             H           T
                    //      [o o o . . . . . . o o o o o I A]
                    //       M M M                         M

                    // copy elements up to new head
                    self.copy(1, 0, self.head);

                    // copy last element into empty spot at bottom of buffer
                    self.copy(0, self.cap() - 1, 1);

                    // move elements from idx to end forward not including ^ element
                    self.copy(idx + 1, idx, self.cap() - 1 - idx);

                    self.head += 1;
                }
            }
            (false, true, false) if idx == 0 => {
                unsafe {
                    // discontiguous, insert is closer to tail, head section,
                    // and is at index zero in the internal buffer:
                    //
                    //       I                   H     T
                    //      [A o o o o o o o o o . . . o o o]
                    //
                    //                           H   T
                    //      [A o o o o o o o o o . . o o o I]
                    //                               M M M

                    // copy elements up to new tail
                    self.copy(self.tail - 1, self.tail, self.cap() - self.tail);

                    // copy last element into empty spot at bottom of buffer
                    self.copy(self.cap() - 1, 0, 1);

                    self.tail -= 1;
                }
            }
            (false, true, false) => {
                unsafe {
                    // discontiguous, insert closer to tail, head section:
                    //
                    //             I             H     T
                    //      [o o o A o o o o o o . . . o o o]
                    //
                    //                           H   T
                    //      [o o I A o o o o o o . . o o o o]
                    //       M M                     M M M M

                    // copy elements up to new tail
                    self.copy(self.tail - 1, self.tail, self.cap() - self.tail);

                    // copy last element into empty spot at bottom of buffer
                    self.copy(self.cap() - 1, 0, 1);

                    // move elements from idx-1 to end forward not including ^ element
                    self.copy(0, 1, idx - 1);

                    self.tail -= 1;
                }
            }
            (false, false, false) => {
                unsafe {
                    // discontiguous, insert closer to head, head section:
                    //
                    //               I     H           T
                    //      [o o o o A o o . . . . . . o o o]
                    //
                    //                     H           T
                    //      [o o o o I A o o . . . . . o o o]
                    //                 M M M

                    self.copy(idx + 1, idx, self.head - idx);
                    self.head += 1;
                }
            }
        }

        // tail might've been changed so we need to recalculate
        let new_idx = self.wrap_add(self.tail, index);
        unsafe {
            self.buffer_write(new_idx, value);
        }
    }

    /// Removes and returns the element at `index` from the `VecDeque`.
    /// Whichever end is closer to the removal point will be moved to make
    /// room, and all the affected elements will be moved to new positions.
    /// Returns `None` if `index` is out of bounds.
    ///
    /// Element at index 0 is the front of the queue.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf = VecDeque::new();
    /// buf.push_back(1);
    /// buf.push_back(2);
    /// buf.push_back(3);
    ///
    /// assert_eq!(buf.remove(1), Some(2));
    /// assert_eq!(buf.get(1), Some(&3));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if self.is_empty() || self.len() <= index {
            return None;
        }

        // There are three main cases:
        //  Elements are contiguous
        //  Elements are discontiguous and the removal is in the tail section
        //  Elements are discontiguous and the removal is in the head section
        //      - special case when elements are technically contiguous,
        //        but self.head = 0
        //
        // For each of those there are two more cases:
        //  Insert is closer to tail
        //  Insert is closer to head
        //
        // Key: H - self.head
        //      T - self.tail
        //      o - Valid element
        //      x - Element marked for removal
        //      R - Indicates element that is being removed
        //      M - Indicates element was moved

        let idx = self.wrap_add(self.tail, index);

        let elem = unsafe { Some(self.buffer_read(idx)) };

        let distance_to_tail = index;
        let distance_to_head = self.len() - index;

        let contiguous = self.is_contiguous();

        match (contiguous, distance_to_tail <= distance_to_head, idx >= self.tail) {
            (true, true, _) => {
                unsafe {
                    // contiguous, remove closer to tail:
                    //
                    //             T   R         H
                    //      [. . . o o x o o o o . . . . . .]
                    //
                    //               T           H
                    //      [. . . . o o o o o o . . . . . .]
                    //               M M

                    self.copy(self.tail + 1, self.tail, index);
                    self.tail += 1;
                }
            }
            (true, false, _) => {
                unsafe {
                    // contiguous, remove closer to head:
                    //
                    //             T       R     H
                    //      [. . . o o o o x o o . . . . . .]
                    //
                    //             T           H
                    //      [. . . o o o o o o . . . . . . .]
                    //                     M M

                    self.copy(idx, idx + 1, self.head - idx - 1);
                    self.head -= 1;
                }
            }
            (false, true, true) => {
                unsafe {
                    // discontiguous, remove closer to tail, tail section:
                    //
                    //                   H         T   R
                    //      [o o o o o o . . . . . o o x o o]
                    //
                    //                   H           T
                    //      [o o o o o o . . . . . . o o o o]
                    //                               M M

                    self.copy(self.tail + 1, self.tail, index);
                    self.tail = self.wrap_add(self.tail, 1);
                }
            }
            (false, false, false) => {
                unsafe {
                    // discontiguous, remove closer to head, head section:
                    //
                    //               R     H           T
                    //      [o o o o x o o . . . . . . o o o]
                    //
                    //                   H             T
                    //      [o o o o o o . . . . . . . o o o]
                    //               M M

                    self.copy(idx, idx + 1, self.head - idx - 1);
                    self.head -= 1;
                }
            }
            (false, false, true) => {
                unsafe {
                    // discontiguous, remove closer to head, tail section:
                    //
                    //             H           T         R
                    //      [o o o . . . . . . o o o o o x o]
                    //
                    //           H             T
                    //      [o o . . . . . . . o o o o o o o]
                    //       M M                         M M
                    //
                    // or quasi-discontiguous, remove next to head, tail section:
                    //
                    //       H                 T         R
                    //      [. . . . . . . . . o o o o o x o]
                    //
                    //                         T           H
                    //      [. . . . . . . . . o o o o o o .]
                    //                                   M

                    // draw in elements in the tail section
                    self.copy(idx, idx + 1, self.cap() - idx - 1);

                    // Prevents underflow.
                    if self.head != 0 {
                        // copy first element into empty spot
                        self.copy(self.cap() - 1, 0, 1);

                        // move elements in the head section backwards
                        self.copy(0, 1, self.head - 1);
                    }

                    self.head = self.wrap_sub(self.head, 1);
                }
            }
            (false, true, false) => {
                unsafe {
                    // discontiguous, remove closer to tail, head section:
                    //
                    //           R               H     T
                    //      [o o x o o o o o o o . . . o o o]
                    //
                    //                           H       T
                    //      [o o o o o o o o o o . . . . o o]
                    //       M M M                       M M

                    // draw in elements up to idx
                    self.copy(1, 0, idx);

                    // copy last element into empty spot
                    self.copy(0, self.cap() - 1, 1);

                    // move elements from tail to end forward, excluding the last one
                    self.copy(self.tail + 1, self.tail, self.cap() - self.tail - 1);

                    self.tail = self.wrap_add(self.tail, 1);
                }
            }
        }

        return elem;
    }

    /// Splits the collection into two at the given index.
    ///
    /// Returns a newly allocated `Self`. `self` contains elements `[0, at)`,
    /// and the returned `Self` contains elements `[at, len)`.
    ///
    /// Note that the capacity of `self` does not change.
    ///
    /// Element at index 0 is the front of the queue.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf: VecDeque<_> = vec![1,2,3].into_iter().collect();
    /// let buf2 = buf.split_off(1);
    /// // buf = [1], buf2 = [2, 3]
    /// assert_eq!(buf.len(), 1);
    /// assert_eq!(buf2.len(), 2);
    /// ```
    #[inline]
    #[stable(feature = "split_off", since = "1.4.0")]
    pub fn split_off(&mut self, at: usize) -> Self {
        let len = self.len();
        assert!(at <= len, "`at` out of bounds");

        let other_len = len - at;
        let mut other = VecDeque::with_capacity(other_len);

        unsafe {
            let (first_half, second_half) = self.as_slices();

            let first_len = first_half.len();
            let second_len = second_half.len();
            if at < first_len {
                // `at` lies in the first half.
                let amount_in_first = first_len - at;

                ptr::copy_nonoverlapping(first_half.as_ptr().offset(at as isize),
                                         other.ptr(),
                                         amount_in_first);

                // just take all of the second half.
                ptr::copy_nonoverlapping(second_half.as_ptr(),
                                         other.ptr().offset(amount_in_first as isize),
                                         second_len);
            } else {
                // `at` lies in the second half, need to factor in the elements we skipped
                // in the first half.
                let offset = at - first_len;
                let amount_in_second = second_len - offset;
                ptr::copy_nonoverlapping(second_half.as_ptr().offset(offset as isize),
                                         other.ptr(),
                                         amount_in_second);
            }
        }

        // Cleanup where the ends of the buffers are
        self.head = self.wrap_sub(self.head, other_len);
        other.head = other.wrap_index(other_len);

        other
    }

    /// Moves all the elements of `other` into `Self`, leaving `other` empty.
    ///
    /// # Panics
    ///
    /// Panics if the new number of elements in self overflows a `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf: VecDeque<_> = vec![1, 2, 3].into_iter().collect();
    /// let mut buf2: VecDeque<_> = vec![4, 5, 6].into_iter().collect();
    /// buf.append(&mut buf2);
    /// assert_eq!(buf.len(), 6);
    /// assert_eq!(buf2.len(), 0);
    /// ```
    #[inline]
    #[stable(feature = "append", since = "1.4.0")]
    pub fn append(&mut self, other: &mut Self) {
        // naive impl
        self.extend(other.drain(..));
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns false.
    /// This method operates in place and preserves the order of the retained
    /// elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let mut buf = VecDeque::new();
    /// buf.extend(1..5);
    /// buf.retain(|&x| x%2 == 0);
    ///
    /// let v: Vec<_> = buf.into_iter().collect();
    /// assert_eq!(&v[..], &[2, 4]);
    /// ```
    #[stable(feature = "vec_deque_retain", since = "1.4.0")]
    pub fn retain<F>(&mut self, mut f: F)
        where F: FnMut(&T) -> bool
    {
        let len = self.len();
        let mut del = 0;
        for i in 0..len {
            if !f(&self[i]) {
                del += 1;
            } else if del > 0 {
                self.swap(i - del, i);
            }
        }
        if del > 0 {
            self.truncate(len - del);
        }
    }
}

impl<T: Clone> VecDeque<T> {
    /// Modifies the `VecDeque` in-place so that `len()` is equal to new_len,
    /// either by removing excess elements or by appending copies of a value to the back.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(deque_extras)]
    ///
    /// use std::collections::VecDeque;
    ///
    /// let mut buf = VecDeque::new();
    /// buf.push_back(5);
    /// buf.push_back(10);
    /// buf.push_back(15);
    /// buf.resize(2, 0);
    /// buf.resize(6, 20);
    /// for (a, b) in [5, 10, 20, 20, 20, 20].iter().zip(&buf) {
    ///     assert_eq!(a, b);
    /// }
    /// ```
    #[unstable(feature = "deque_extras",
               reason = "matches collection reform specification; waiting on panic semantics",
               issue = "27788")]
    pub fn resize(&mut self, new_len: usize, value: T) {
        let len = self.len();

        if new_len > len {
            self.extend(repeat(value).take(new_len - len))
        } else {
            self.truncate(new_len);
        }
    }
}

/// Returns the index in the underlying buffer for a given logical element index.
#[inline]
fn wrap_index(index: usize, size: usize) -> usize {
    // size is always a power of 2
    debug_assert!(size.is_power_of_two());
    index & (size - 1)
}

/// Returns the two slices that cover the VecDeque's valid range
trait RingSlices: Sized {
    fn slice(self, from: usize, to: usize) -> Self;
    fn split_at(self, i: usize) -> (Self, Self);

    fn ring_slices(buf: Self, head: usize, tail: usize) -> (Self, Self) {
        let contiguous = tail <= head;
        if contiguous {
            let (empty, buf) = buf.split_at(0);
            (buf.slice(tail, head), empty)
        } else {
            let (mid, right) = buf.split_at(tail);
            let (left, _) = mid.split_at(head);
            (right, left)
        }
    }
}

impl<'a, T> RingSlices for &'a [T] {
    fn slice(self, from: usize, to: usize) -> Self {
        &self[from..to]
    }
    fn split_at(self, i: usize) -> (Self, Self) {
        (*self).split_at(i)
    }
}

impl<'a, T> RingSlices for &'a mut [T] {
    fn slice(self, from: usize, to: usize) -> Self {
        &mut self[from..to]
    }
    fn split_at(self, i: usize) -> (Self, Self) {
        (*self).split_at_mut(i)
    }
}

/// Calculate the number of elements left to be read in the buffer
#[inline]
fn count(tail: usize, head: usize, size: usize) -> usize {
    // size is always a power of 2
    (head.wrapping_sub(tail)) & (size - 1)
}

/// `VecDeque` iterator.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    ring: &'a [T],
    tail: usize,
    head: usize,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Iter<'a, T> {
        Iter {
            ring: self.ring,
            tail: self.tail,
            head: self.head,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.tail == self.head {
            return None;
        }
        let tail = self.tail;
        self.tail = wrap_index(self.tail.wrapping_add(1), self.ring.len());
        unsafe { Some(self.ring.get_unchecked(tail)) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = count(self.tail, self.head, self.ring.len());
        (len, Some(len))
    }

    fn fold<Acc, F>(self, mut accum: Acc, mut f: F) -> Acc
        where F: FnMut(Acc, Self::Item) -> Acc
    {
        let (front, back) = RingSlices::ring_slices(self.ring, self.head, self.tail);
        accum = front.iter().fold(accum, &mut f);
        back.iter().fold(accum, &mut f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        if self.tail == self.head {
            return None;
        }
        self.head = wrap_index(self.head.wrapping_sub(1), self.ring.len());
        unsafe { Some(self.ring.get_unchecked(self.head)) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for Iter<'a, T> {
    fn is_empty(&self) -> bool {
        self.head == self.tail
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<'a, T> FusedIterator for Iter<'a, T> {}


/// `VecDeque` mutable iterator.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, T: 'a> {
    ring: &'a mut [T],
    tail: usize,
    head: usize,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        if self.tail == self.head {
            return None;
        }
        let tail = self.tail;
        self.tail = wrap_index(self.tail.wrapping_add(1), self.ring.len());

        unsafe {
            let elem = self.ring.get_unchecked_mut(tail);
            Some(&mut *(elem as *mut _))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = count(self.tail, self.head, self.ring.len());
        (len, Some(len))
    }

    fn fold<Acc, F>(self, mut accum: Acc, mut f: F) -> Acc
        where F: FnMut(Acc, Self::Item) -> Acc
    {
        let (front, back) = RingSlices::ring_slices(self.ring, self.head, self.tail);
        accum = front.iter_mut().fold(accum, &mut f);
        back.iter_mut().fold(accum, &mut f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut T> {
        if self.tail == self.head {
            return None;
        }
        self.head = wrap_index(self.head.wrapping_sub(1), self.ring.len());

        unsafe {
            let elem = self.ring.get_unchecked_mut(self.head);
            Some(&mut *(elem as *mut _))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for IterMut<'a, T> {
    fn is_empty(&self) -> bool {
        self.head == self.tail
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<'a, T> FusedIterator for IterMut<'a, T> {}

/// A by-value VecDeque iterator
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<T> {
    inner: VecDeque<T>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.pop_front()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len();
        (len, Some(len))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.inner.pop_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for IntoIter<T> {
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<T> FusedIterator for IntoIter<T> {}

/// A draining VecDeque iterator
#[stable(feature = "drain", since = "1.6.0")]
pub struct Drain<'a, T: 'a> {
    after_tail: usize,
    after_head: usize,
    iter: Iter<'a, T>,
    deque: Shared<VecDeque<T>>,
}

#[stable(feature = "drain", since = "1.6.0")]
unsafe impl<'a, T: Sync> Sync for Drain<'a, T> {}
#[stable(feature = "drain", since = "1.6.0")]
unsafe impl<'a, T: Send> Send for Drain<'a, T> {}

#[stable(feature = "drain", since = "1.6.0")]
impl<'a, T: 'a> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        for _ in self.by_ref() {}

        let source_deque = unsafe { &mut **self.deque };

        // T = source_deque_tail; H = source_deque_head; t = drain_tail; h = drain_head
        //
        //        T   t   h   H
        // [. . . o o x x o o . . .]
        //
        let orig_tail = source_deque.tail;
        let drain_tail = source_deque.head;
        let drain_head = self.after_tail;
        let orig_head = self.after_head;

        let tail_len = count(orig_tail, drain_tail, source_deque.cap());
        let head_len = count(drain_head, orig_head, source_deque.cap());

        // Restore the original head value
        source_deque.head = orig_head;

        match (tail_len, head_len) {
            (0, 0) => {
                source_deque.head = 0;
                source_deque.tail = 0;
            }
            (0, _) => {
                source_deque.tail = drain_head;
            }
            (_, 0) => {
                source_deque.head = drain_tail;
            }
            _ => unsafe {
                if tail_len <= head_len {
                    source_deque.tail = source_deque.wrap_sub(drain_head, tail_len);
                    source_deque.wrap_copy(source_deque.tail, orig_tail, tail_len);
                } else {
                    source_deque.head = source_deque.wrap_add(drain_tail, head_len);
                    source_deque.wrap_copy(drain_tail, drain_head, head_len);
                }
            },
        }
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<'a, T: 'a> Iterator for Drain<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter.next().map(|elt| unsafe { ptr::read(elt) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<'a, T: 'a> DoubleEndedIterator for Drain<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back().map(|elt| unsafe { ptr::read(elt) })
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<'a, T: 'a> ExactSizeIterator for Drain<'a, T> {}

#[unstable(feature = "fused", issue = "35602")]
impl<'a, T: 'a> FusedIterator for Drain<'a, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: PartialEq> PartialEq for VecDeque<A> {
    fn eq(&self, other: &VecDeque<A>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        let (sa, sb) = self.as_slices();
        let (oa, ob) = other.as_slices();
        if sa.len() == oa.len() {
            sa == oa && sb == ob
        } else if sa.len() < oa.len() {
            // Always divisible in three sections, for example:
            // self:  [a b c|d e f]
            // other: [0 1 2 3|4 5]
            // front = 3, mid = 1,
            // [a b c] == [0 1 2] && [d] == [3] && [e f] == [4 5]
            let front = sa.len();
            let mid = oa.len() - front;

            let (oa_front, oa_mid) = oa.split_at(front);
            let (sb_mid, sb_back) = sb.split_at(mid);
            debug_assert_eq!(sa.len(), oa_front.len());
            debug_assert_eq!(sb_mid.len(), oa_mid.len());
            debug_assert_eq!(sb_back.len(), ob.len());
            sa == oa_front && sb_mid == oa_mid && sb_back == ob
        } else {
            let front = oa.len();
            let mid = sa.len() - front;

            let (sa_front, sa_mid) = sa.split_at(front);
            let (ob_mid, ob_back) = ob.split_at(mid);
            debug_assert_eq!(sa_front.len(), oa.len());
            debug_assert_eq!(sa_mid.len(), ob_mid.len());
            debug_assert_eq!(sb.len(), ob_back.len());
            sa_front == oa && sa_mid == ob_mid && sb == ob_back
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Eq> Eq for VecDeque<A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: PartialOrd> PartialOrd for VecDeque<A> {
    fn partial_cmp(&self, other: &VecDeque<A>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Ord> Ord for VecDeque<A> {
    #[inline]
    fn cmp(&self, other: &VecDeque<A>) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Hash> Hash for VecDeque<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        let (a, b) = self.as_slices();
        Hash::hash_slice(a, state);
        Hash::hash_slice(b, state);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> Index<usize> for VecDeque<A> {
    type Output = A;

    #[inline]
    fn index(&self, index: usize) -> &A {
        self.get(index).expect("Out of bounds access")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> IndexMut<usize> for VecDeque<A> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut A {
        self.get_mut(index).expect("Out of bounds access")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> FromIterator<A> for VecDeque<A> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> VecDeque<A> {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();
        let mut deq = VecDeque::with_capacity(lower);
        deq.extend(iterator);
        deq
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> IntoIterator for VecDeque<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Consumes the list into a front-to-back iterator yielding elements by
    /// value.
    fn into_iter(self) -> IntoIter<T> {
        IntoIter { inner: self }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a VecDeque<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a mut VecDeque<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(mut self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> Extend<A> for VecDeque<A> {
    fn extend<T: IntoIterator<Item = A>>(&mut self, iter: T) {
        for elt in iter {
            self.push_back(elt);
        }
    }
}

#[stable(feature = "extend_ref", since = "1.2.0")]
impl<'a, T: 'a + Copy> Extend<&'a T> for VecDeque<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Debug> fmt::Debug for VecDeque<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

#[stable(feature = "vecdeque_vec_conversions", since = "1.10.0")]
impl<T> From<Vec<T>> for VecDeque<T> {
    fn from(mut other: Vec<T>) -> Self {
        unsafe {
            let other_buf = other.as_mut_ptr();
            let mut buf = RawVec::from_raw_parts(other_buf, other.capacity());
            let len = other.len();
            mem::forget(other);

            // We need to extend the buf if it's not a power of two, too small
            // or doesn't have at least one free space
            if !buf.cap().is_power_of_two() || (buf.cap() < (MINIMUM_CAPACITY + 1)) ||
               (buf.cap() == len) {
                let cap = cmp::max(buf.cap() + 1, MINIMUM_CAPACITY + 1).next_power_of_two();
                buf.reserve_exact(len, cap - len);
            }

            VecDeque {
                tail: 0,
                head: len,
                buf: buf,
            }
        }
    }
}

#[stable(feature = "vecdeque_vec_conversions", since = "1.10.0")]
impl<T> From<VecDeque<T>> for Vec<T> {
    fn from(other: VecDeque<T>) -> Self {
        unsafe {
            let buf = other.buf.ptr();
            let len = other.len();
            let tail = other.tail;
            let head = other.head;
            let cap = other.cap();

            // Need to move the ring to the front of the buffer, as vec will expect this.
            if other.is_contiguous() {
                ptr::copy(buf.offset(tail as isize), buf, len);
            } else {
                if (tail - head) >= cmp::min((cap - tail), head) {
                    // There is enough free space in the centre for the shortest block so we can
                    // do this in at most three copy moves.
                    if (cap - tail) > head {
                        // right hand block is the long one; move that enough for the left
                        ptr::copy(buf.offset(tail as isize),
                                  buf.offset((tail - head) as isize),
                                  cap - tail);
                        // copy left in the end
                        ptr::copy(buf, buf.offset((cap - head) as isize), head);
                        // shift the new thing to the start
                        ptr::copy(buf.offset((tail - head) as isize), buf, len);
                    } else {
                        // left hand block is the long one, we can do it in two!
                        ptr::copy(buf, buf.offset((cap - tail) as isize), head);
                        ptr::copy(buf.offset(tail as isize), buf, cap - tail);
                    }
                } else {
                    // Need to use N swaps to move the ring
                    // We can use the space at the end of the ring as a temp store

                    let mut left_edge: usize = 0;
                    let mut right_edge: usize = tail;

                    // The general problem looks like this
                    // GHIJKLM...ABCDEF - before any swaps
                    // ABCDEFM...GHIJKL - after 1 pass of swaps
                    // ABCDEFGHIJM...KL - swap until the left edge reaches the temp store
                    //                  - then restart the algorithm with a new (smaller) store
                    // Sometimes the temp store is reached when the right edge is at the end
                    // of the buffer - this means we've hit the right order with fewer swaps!
                    // E.g
                    // EF..ABCD
                    // ABCDEF.. - after four only swaps we've finished

                    while left_edge < len && right_edge != cap {
                        let mut right_offset = 0;
                        for i in left_edge..right_edge {
                            right_offset = (i - left_edge) % (cap - right_edge);
                            let src: isize = (right_edge + right_offset) as isize;
                            ptr::swap(buf.offset(i as isize), buf.offset(src));
                        }
                        let n_ops = right_edge - left_edge;
                        left_edge += n_ops;
                        right_edge += right_offset + 1;

                    }
                }

            }
            let out = Vec::from_raw_parts(buf, len, cap);
            mem::forget(other);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use test;

    use super::VecDeque;

    #[bench]
    fn bench_push_back_100(b: &mut test::Bencher) {
        let mut deq = VecDeque::with_capacity(101);
        b.iter(|| {
            for i in 0..100 {
                deq.push_back(i);
            }
            deq.head = 0;
            deq.tail = 0;
        })
    }

    #[bench]
    fn bench_push_front_100(b: &mut test::Bencher) {
        let mut deq = VecDeque::with_capacity(101);
        b.iter(|| {
            for i in 0..100 {
                deq.push_front(i);
            }
            deq.head = 0;
            deq.tail = 0;
        })
    }

    #[bench]
    fn bench_pop_back_100(b: &mut test::Bencher) {
        let mut deq = VecDeque::<i32>::with_capacity(101);

        b.iter(|| {
            deq.head = 100;
            deq.tail = 0;
            while !deq.is_empty() {
                test::black_box(deq.pop_back());
            }
        })
    }

    #[bench]
    fn bench_pop_front_100(b: &mut test::Bencher) {
        let mut deq = VecDeque::<i32>::with_capacity(101);

        b.iter(|| {
            deq.head = 100;
            deq.tail = 0;
            while !deq.is_empty() {
                test::black_box(deq.pop_front());
            }
        })
    }

    #[test]
    fn test_swap_front_back_remove() {
        fn test(back: bool) {
            // This test checks that every single combination of tail position and length is tested.
            // Capacity 15 should be large enough to cover every case.
            let mut tester = VecDeque::with_capacity(15);
            let usable_cap = tester.capacity();
            let final_len = usable_cap / 2;

            for len in 0..final_len {
                let expected = if back {
                    (0..len).collect()
                } else {
                    (0..len).rev().collect()
                };
                for tail_pos in 0..usable_cap {
                    tester.tail = tail_pos;
                    tester.head = tail_pos;
                    if back {
                        for i in 0..len * 2 {
                            tester.push_front(i);
                        }
                        for i in 0..len {
                            assert_eq!(tester.swap_remove_back(i), Some(len * 2 - 1 - i));
                        }
                    } else {
                        for i in 0..len * 2 {
                            tester.push_back(i);
                        }
                        for i in 0..len {
                            let idx = tester.len() - 1 - i;
                            assert_eq!(tester.swap_remove_front(idx), Some(len * 2 - 1 - i));
                        }
                    }
                    assert!(tester.tail < tester.cap());
                    assert!(tester.head < tester.cap());
                    assert_eq!(tester, expected);
                }
            }
        }
        test(true);
        test(false);
    }

    #[test]
    fn test_insert() {
        // This test checks that every single combination of tail position, length, and
        // insertion position is tested. Capacity 15 should be large enough to cover every case.

        let mut tester = VecDeque::with_capacity(15);
        // can't guarantee we got 15, so have to get what we got.
        // 15 would be great, but we will definitely get 2^k - 1, for k >= 4, or else
        // this test isn't covering what it wants to
        let cap = tester.capacity();


        // len is the length *after* insertion
        for len in 1..cap {
            // 0, 1, 2, .., len - 1
            let expected = (0..).take(len).collect();
            for tail_pos in 0..cap {
                for to_insert in 0..len {
                    tester.tail = tail_pos;
                    tester.head = tail_pos;
                    for i in 0..len {
                        if i != to_insert {
                            tester.push_back(i);
                        }
                    }
                    tester.insert(to_insert, to_insert);
                    assert!(tester.tail < tester.cap());
                    assert!(tester.head < tester.cap());
                    assert_eq!(tester, expected);
                }
            }
        }
    }

    #[test]
    fn test_remove() {
        // This test checks that every single combination of tail position, length, and
        // removal position is tested. Capacity 15 should be large enough to cover every case.

        let mut tester = VecDeque::with_capacity(15);
        // can't guarantee we got 15, so have to get what we got.
        // 15 would be great, but we will definitely get 2^k - 1, for k >= 4, or else
        // this test isn't covering what it wants to
        let cap = tester.capacity();

        // len is the length *after* removal
        for len in 0..cap - 1 {
            // 0, 1, 2, .., len - 1
            let expected = (0..).take(len).collect();
            for tail_pos in 0..cap {
                for to_remove in 0..len + 1 {
                    tester.tail = tail_pos;
                    tester.head = tail_pos;
                    for i in 0..len {
                        if i == to_remove {
                            tester.push_back(1234);
                        }
                        tester.push_back(i);
                    }
                    if to_remove == len {
                        tester.push_back(1234);
                    }
                    tester.remove(to_remove);
                    assert!(tester.tail < tester.cap());
                    assert!(tester.head < tester.cap());
                    assert_eq!(tester, expected);
                }
            }
        }
    }

    #[test]
    fn test_drain() {
        let mut tester: VecDeque<usize> = VecDeque::with_capacity(7);

        let cap = tester.capacity();
        for len in 0..cap + 1 {
            for tail in 0..cap + 1 {
                for drain_start in 0..len + 1 {
                    for drain_end in drain_start..len + 1 {
                        tester.tail = tail;
                        tester.head = tail;
                        for i in 0..len {
                            tester.push_back(i);
                        }

                        // Check that we drain the correct values
                        let drained: VecDeque<_> = tester.drain(drain_start..drain_end).collect();
                        let drained_expected: VecDeque<_> = (drain_start..drain_end).collect();
                        assert_eq!(drained, drained_expected);

                        // We shouldn't have changed the capacity or made the
                        // head or tail out of bounds
                        assert_eq!(tester.capacity(), cap);
                        assert!(tester.tail < tester.cap());
                        assert!(tester.head < tester.cap());

                        // We should see the correct values in the VecDeque
                        let expected: VecDeque<_> = (0..drain_start)
                            .chain(drain_end..len)
                            .collect();
                        assert_eq!(expected, tester);
                    }
                }
            }
        }
    }

    #[test]
    fn test_shrink_to_fit() {
        // This test checks that every single combination of head and tail position,
        // is tested. Capacity 15 should be large enough to cover every case.

        let mut tester = VecDeque::with_capacity(15);
        // can't guarantee we got 15, so have to get what we got.
        // 15 would be great, but we will definitely get 2^k - 1, for k >= 4, or else
        // this test isn't covering what it wants to
        let cap = tester.capacity();
        tester.reserve(63);
        let max_cap = tester.capacity();

        for len in 0..cap + 1 {
            // 0, 1, 2, .., len - 1
            let expected = (0..).take(len).collect();
            for tail_pos in 0..max_cap + 1 {
                tester.tail = tail_pos;
                tester.head = tail_pos;
                tester.reserve(63);
                for i in 0..len {
                    tester.push_back(i);
                }
                tester.shrink_to_fit();
                assert!(tester.capacity() <= cap);
                assert!(tester.tail < tester.cap());
                assert!(tester.head < tester.cap());
                assert_eq!(tester, expected);
            }
        }
    }

    #[test]
    fn test_split_off() {
        // This test checks that every single combination of tail position, length, and
        // split position is tested. Capacity 15 should be large enough to cover every case.

        let mut tester = VecDeque::with_capacity(15);
        // can't guarantee we got 15, so have to get what we got.
        // 15 would be great, but we will definitely get 2^k - 1, for k >= 4, or else
        // this test isn't covering what it wants to
        let cap = tester.capacity();

        // len is the length *before* splitting
        for len in 0..cap {
            // index to split at
            for at in 0..len + 1 {
                // 0, 1, 2, .., at - 1 (may be empty)
                let expected_self = (0..).take(at).collect();
                // at, at + 1, .., len - 1 (may be empty)
                let expected_other = (at..).take(len - at).collect();

                for tail_pos in 0..cap {
                    tester.tail = tail_pos;
                    tester.head = tail_pos;
                    for i in 0..len {
                        tester.push_back(i);
                    }
                    let result = tester.split_off(at);
                    assert!(tester.tail < tester.cap());
                    assert!(tester.head < tester.cap());
                    assert!(result.tail < result.cap());
                    assert!(result.head < result.cap());
                    assert_eq!(tester, expected_self);
                    assert_eq!(result, expected_other);
                }
            }
        }
    }

    #[test]
    fn test_from_vec() {
        use super::super::vec::Vec;
        for cap in 0..35 {
            for len in 0..cap + 1 {
                let mut vec = Vec::with_capacity(cap);
                vec.extend(0..len);

                let vd = VecDeque::from(vec.clone());
                assert!(vd.cap().is_power_of_two());
                assert_eq!(vd.len(), vec.len());
                assert!(vd.into_iter().eq(vec));
            }
        }
    }

    #[test]
    fn test_vec_from_vecdeque() {
        use super::super::vec::Vec;

        fn create_vec_and_test_convert(cap: usize, offset: usize, len: usize) {
            let mut vd = VecDeque::with_capacity(cap);
            for _ in 0..offset {
                vd.push_back(0);
                vd.pop_front();
            }
            vd.extend(0..len);

            let vec: Vec<_> = Vec::from(vd.clone());
            assert_eq!(vec.len(), vd.len());
            assert!(vec.into_iter().eq(vd));
        }

        for cap_pwr in 0..7 {
            // Make capacity as a (2^x)-1, so that the ring size is 2^x
            let cap = (2i32.pow(cap_pwr) - 1) as usize;

            // In these cases there is enough free space to solve it with copies
            for len in 0..((cap + 1) / 2) {
                // Test contiguous cases
                for offset in 0..(cap - len) {
                    create_vec_and_test_convert(cap, offset, len)
                }

                // Test cases where block at end of buffer is bigger than block at start
                for offset in (cap - len)..(cap - (len / 2)) {
                    create_vec_and_test_convert(cap, offset, len)
                }

                // Test cases where block at start of buffer is bigger than block at end
                for offset in (cap - (len / 2))..cap {
                    create_vec_and_test_convert(cap, offset, len)
                }
            }

            // Now there's not (necessarily) space to straighten the ring with simple copies,
            // the ring will use swapping when:
            // (cap + 1 - offset) > (cap + 1 - len) && (len - (cap + 1 - offset)) > (cap + 1 - len))
            //  right block size  >   free space    &&      left block size       >    free space
            for len in ((cap + 1) / 2)..cap {
                // Test contiguous cases
                for offset in 0..(cap - len) {
                    create_vec_and_test_convert(cap, offset, len)
                }

                // Test cases where block at end of buffer is bigger than block at start
                for offset in (cap - len)..(cap - (len / 2)) {
                    create_vec_and_test_convert(cap, offset, len)
                }

                // Test cases where block at start of buffer is bigger than block at end
                for offset in (cap - (len / 2))..cap {
                    create_vec_and_test_convert(cap, offset, len)
                }
            }
        }
    }
}
