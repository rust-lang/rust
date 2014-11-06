// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This crate implements a double-ended queue with `O(1)` amortized inserts and removals from both
//! ends of the container. It also has `O(1)` indexing like a vector. The contained elements are
//! not required to be copyable, and the queue will be sendable if the contained type is sendable.
//! Its interface `Deque` is defined in `collections`.

use core::prelude::*;

use core::default::Default;
use core::fmt;
use core::iter;
use core::raw::Slice as RawSlice;
use core::ptr;
use core::kinds::marker;
use core::mem;
use core::num::{Int, UnsignedInt};

use std::hash::{Writer, Hash};
use std::cmp;

use alloc::heap;

static INITIAL_CAPACITY: uint = 8u; // 2^3
static MINIMUM_CAPACITY: uint = 2u;

// FIXME(conventions): implement shrink_to_fit. Awkward with the current design, but it should
// be scrapped anyway. Defer to rewrite?
// FIXME(conventions): implement into_iter


/// `RingBuf` is a circular buffer that implements `Deque`.
pub struct RingBuf<T> {
    // tail and head are pointers into the buffer. Tail always points
    // to the first element that could be read, Head always points
    // to where data should be written.
    // If tail == head the buffer is empty. The length of the ringbuf
    // is defined as the distance between the two.

    tail: uint,
    head: uint,
    cap: uint,
    ptr: *mut T
}

impl<T: Clone> Clone for RingBuf<T> {
    fn clone(&self) -> RingBuf<T> {
        self.iter().map(|t| t.clone()).collect()
    }
}

#[unsafe_destructor]
impl<T> Drop for RingBuf<T> {
    fn drop(&mut self) {
        self.clear();
        unsafe {
            if mem::size_of::<T>() != 0 {
                heap::deallocate(self.ptr as *mut u8,
                                 self.cap * mem::size_of::<T>(),
                                 mem::min_align_of::<T>())
            }
        }
    }
}

impl<T> Default for RingBuf<T> {
    #[inline]
    fn default() -> RingBuf<T> { RingBuf::new() }
}

impl<T> RingBuf<T> {
    /// Turn ptr into a slice
    #[inline]
    unsafe fn buffer_as_slice(&self) -> &[T] {
        mem::transmute(RawSlice { data: self.ptr as *const T, len: self.cap })
    }

    /// Moves an element out of the buffer
    #[inline]
    unsafe fn buffer_read(&mut self, off: uint) -> T {
        ptr::read(self.ptr.offset(off as int) as *const T)
    }

    /// Writes an element into the buffer, moving it.
    #[inline]
    unsafe fn buffer_write(&mut self, off: uint, t: T) {
        ptr::write(self.ptr.offset(off as int), t);
    }

    /// Returns true iff the buffer is at capacity
    #[inline]
    fn is_full(&self) -> bool { self.cap - self.len() == 1 }

    /// Returns the index in the underlying buffer for a given logical element index.
    #[inline]
    fn wrap_index(&self, idx: uint) -> uint { wrap_index(idx, self.cap) }
}

impl<T> RingBuf<T> {
    /// Creates an empty `RingBuf`.
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn new() -> RingBuf<T> {
        RingBuf::with_capacity(INITIAL_CAPACITY)
    }

    /// Creates an empty `RingBuf` with space for at least `n` elements.
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn with_capacity(n: uint) -> RingBuf<T> {
        // +1 since the ringbuffer always leaves one space empty
        let cap = cmp::max(n + 1, MINIMUM_CAPACITY).next_power_of_two();
        let size = cap.checked_mul(mem::size_of::<T>())
                      .expect("capacity overflow");

        let ptr = if mem::size_of::<T>() != 0 {
            unsafe {
                let ptr = heap::allocate(size, mem::min_align_of::<T>())  as *mut T;;
                if ptr.is_null() { ::alloc::oom() }
                ptr
            }
        } else {
            heap::EMPTY as *mut T
        };

        RingBuf {
            tail: 0,
            head: 0,
            cap: cap,
            ptr: ptr
        }
    }

    /// Retrieves an element in the `RingBuf` by index.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::collections::RingBuf;
    ///
    /// let mut buf = RingBuf::new();
    /// buf.push_back(3i);
    /// buf.push_back(4);
    /// buf.push_back(5);
    /// assert_eq!(buf.get(1).unwrap(), &4);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn get(&self, i: uint) -> Option<&T> {
        if i < self.len() {
            let idx = self.wrap_index(self.tail + i);
            unsafe { Some(&*self.ptr.offset(idx as int)) }
        } else {
            None
        }
    }

    /// Retrieves an element in the `RingBuf` mutably by index.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::collections::RingBuf;
    ///
    /// let mut buf = RingBuf::new();
    /// buf.push_back(3i);
    /// buf.push_back(4);
    /// buf.push_back(5);
    /// match buf.get_mut(1) {
    ///     None => {}
    ///     Some(elem) => {
    ///         *elem = 7;
    ///     }
    /// }
    ///
    /// assert_eq!(buf[1], 7);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn get_mut(&mut self, i: uint) -> Option<&mut T> {
        if i < self.len() {
            let idx = self.wrap_index(self.tail + i);
            unsafe { Some(&mut *self.ptr.offset(idx as int)) }
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
    /// # Example
    ///
    /// ```rust
    /// use std::collections::RingBuf;
    ///
    /// let mut buf = RingBuf::new();
    /// buf.push_back(3i);
    /// buf.push_back(4);
    /// buf.push_back(5);
    /// buf.swap(0, 2);
    /// assert_eq!(buf[0], 5);
    /// assert_eq!(buf[2], 3);
    /// ```
    pub fn swap(&mut self, i: uint, j: uint) {
        assert!(i < self.len());
        assert!(j < self.len());
        let ri = self.wrap_index(self.tail + i);
        let rj = self.wrap_index(self.tail + j);
        unsafe {
            ptr::swap(self.ptr.offset(ri as int), self.ptr.offset(rj as int))
        }
    }

    /// Returns the number of elements the `RingBuf` can hold without
    /// reallocating.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::RingBuf;
    ///
    /// let buf: RingBuf<int> = RingBuf::with_capacity(10);
    /// assert!(buf.capacity() >= 10);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn capacity(&self) -> uint { self.cap - 1 }

    /// Reserves the minimum capacity for exactly `additional` more elements to be inserted in the
    /// given `RingBuf`. Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore
    /// capacity can not be relied upon to be precisely minimal. Prefer `reserve` if future
    /// insertions are expected.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `uint`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::RingBuf;
    ///
    /// let mut buf: RingBuf<int> = vec![1].into_iter().collect();
    /// buf.reserve_exact(10);
    /// assert!(buf.capacity() >= 11);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn reserve_exact(&mut self, additional: uint) {
        self.reserve(additional);
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in the given
    /// `Ringbuf`. The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `uint`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::RingBuf;
    ///
    /// let mut buf: RingBuf<int> = vec![1].into_iter().collect();
    /// buf.reserve(10);
    /// assert!(buf.capacity() >= 11);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn reserve(&mut self, additional: uint) {
        let new_len = self.len() + additional;
        assert!(new_len + 1 > self.len(), "capacity overflow");
        if new_len > self.capacity() {
            let count = (new_len + 1).next_power_of_two();
            assert!(count >= new_len + 1);

            if mem::size_of::<T>() != 0 {
                let old = self.cap * mem::size_of::<T>();
                let new = count.checked_mul(mem::size_of::<T>())
                               .expect("capacity overflow");
                unsafe {
                    self.ptr = heap::reallocate(self.ptr as *mut u8,
                                                old,
                                                new,
                                                mem::min_align_of::<T>()) as *mut T;
                    if self.ptr.is_null() { ::alloc::oom() }
                }
            }

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

            let oldcap = self.cap;
            self.cap = count;

            if self.tail <= self.head { // A
                // Nop
            } else if self.head < oldcap - self.tail { // B
                unsafe {
                    ptr::copy_nonoverlapping_memory(
                        self.ptr.offset(oldcap as int),
                        self.ptr as *const T,
                        self.head
                    );
                }
                self.head += oldcap;
                debug_assert!(self.head > self.tail);
            } else { // C
                unsafe {
                    ptr::copy_nonoverlapping_memory(
                        self.ptr.offset((count - (oldcap - self.tail)) as int),
                        self.ptr.offset(self.tail as int) as *const T,
                        oldcap - self.tail
                    );
                }
                self.tail = count - (oldcap - self.tail);
                debug_assert!(self.head < self.tail);
            }
            debug_assert!(self.head < self.cap);
            debug_assert!(self.tail < self.cap);
            debug_assert!(self.cap.count_ones() == 1);
        }
    }

    /// Returns a front-to-back iterator.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::collections::RingBuf;
    ///
    /// let mut buf = RingBuf::new();
    /// buf.push_back(5i);
    /// buf.push_back(3);
    /// buf.push_back(4);
    /// let b: &[_] = &[&5, &3, &4];
    /// assert_eq!(buf.iter().collect::<Vec<&int>>().as_slice(), b);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn iter(&self) -> Items<T> {
        Items {
            tail: self.tail,
            head: self.head,
            ring: unsafe { self.buffer_as_slice() }
        }
    }

    /// Returns a front-to-back iterator which returns mutable references.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::collections::RingBuf;
    ///
    /// let mut buf = RingBuf::new();
    /// buf.push_back(5i);
    /// buf.push_back(3);
    /// buf.push_back(4);
    /// for num in buf.iter_mut() {
    ///     *num = *num - 2;
    /// }
    /// let b: &[_] = &[&mut 3, &mut 1, &mut 2];
    /// assert_eq!(buf.iter_mut().collect::<Vec<&mut int>>()[], b);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn iter_mut<'a>(&'a mut self) -> MutItems<'a, T> {
        MutItems {
            tail: self.tail,
            head: self.head,
            cap: self.cap,
            ptr: self.ptr,
            marker: marker::ContravariantLifetime::<'a>,
            marker2: marker::NoCopy
        }
    }

    /// Returns the number of elements in the `RingBuf`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::RingBuf;
    ///
    /// let mut v = RingBuf::new();
    /// assert_eq!(v.len(), 0);
    /// v.push_back(1i);
    /// assert_eq!(v.len(), 1);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn len(&self) -> uint { count(self.tail, self.head, self.cap) }

    /// Returns true if the buffer contains no elements
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::RingBuf;
    ///
    /// let mut v = RingBuf::new();
    /// assert!(v.is_empty());
    /// v.push_front(1i);
    /// assert!(!v.is_empty());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Clears the buffer, removing all values.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::RingBuf;
    ///
    /// let mut v = RingBuf::new();
    /// v.push_back(1i);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn clear(&mut self) {
        while self.pop_front().is_some() {}
        self.head = 0;
        self.tail = 0;
    }

    /// Provides a reference to the front element, or `None` if the sequence is
    /// empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::RingBuf;
    ///
    /// let mut d = RingBuf::new();
    /// assert_eq!(d.front(), None);
    ///
    /// d.push_back(1i);
    /// d.push_back(2i);
    /// assert_eq!(d.front(), Some(&1i));
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn front(&self) -> Option<&T> {
        if !self.is_empty() { Some(&self[0]) } else { None }
    }

    /// Provides a mutable reference to the front element, or `None` if the
    /// sequence is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::RingBuf;
    ///
    /// let mut d = RingBuf::new();
    /// assert_eq!(d.front_mut(), None);
    ///
    /// d.push_back(1i);
    /// d.push_back(2i);
    /// match d.front_mut() {
    ///     Some(x) => *x = 9i,
    ///     None => (),
    /// }
    /// assert_eq!(d.front(), Some(&9i));
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn front_mut(&mut self) -> Option<&mut T> {
        if !self.is_empty() { Some(&mut self[0]) } else { None }
    }

    /// Provides a reference to the back element, or `None` if the sequence is
    /// empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::RingBuf;
    ///
    /// let mut d = RingBuf::new();
    /// assert_eq!(d.back(), None);
    ///
    /// d.push_back(1i);
    /// d.push_back(2i);
    /// assert_eq!(d.back(), Some(&2i));
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn back(&self) -> Option<&T> {
        if !self.is_empty() { Some(&self[self.len() - 1]) } else { None }
    }

    /// Provides a mutable reference to the back element, or `None` if the
    /// sequence is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::RingBuf;
    ///
    /// let mut d = RingBuf::new();
    /// assert_eq!(d.back(), None);
    ///
    /// d.push_back(1i);
    /// d.push_back(2i);
    /// match d.back_mut() {
    ///     Some(x) => *x = 9i,
    ///     None => (),
    /// }
    /// assert_eq!(d.back(), Some(&9i));
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn back_mut(&mut self) -> Option<&mut T> {
        let len = self.len();
        if !self.is_empty() { Some(&mut self[len - 1]) } else { None }
    }

    /// Removes the first element and returns it, or `None` if the sequence is
    /// empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::RingBuf;
    ///
    /// let mut d = RingBuf::new();
    /// d.push_back(1i);
    /// d.push_back(2i);
    ///
    /// assert_eq!(d.pop_front(), Some(1i));
    /// assert_eq!(d.pop_front(), Some(2i));
    /// assert_eq!(d.pop_front(), None);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let tail = self.tail;
            self.tail = self.wrap_index(self.tail + 1);
            unsafe { Some(self.buffer_read(tail)) }
        }
    }

    /// Inserts an element first in the sequence.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::RingBuf;
    ///
    /// let mut d = RingBuf::new();
    /// d.push_front(1i);
    /// d.push_front(2i);
    /// assert_eq!(d.front(), Some(&2i));
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn push_front(&mut self, t: T) {
        if self.is_full() {
            self.reserve(1);
            debug_assert!(!self.is_full());
        }

        self.tail = self.wrap_index(self.tail - 1);
        let tail = self.tail;
        unsafe { self.buffer_write(tail, t); }
    }

    /// Deprecated: Renamed to `push_back`.
    #[deprecated = "Renamed to `push_back`"]
    pub fn push(&mut self, t: T) {
        self.push_back(t)
    }

    /// Appends an element to the back of a buffer
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::collections::RingBuf;
    ///
    /// let mut buf = RingBuf::new();
    /// buf.push_back(1i);
    /// buf.push_back(3);
    /// assert_eq!(3, *buf.back().unwrap());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn push_back(&mut self, t: T) {
        if self.is_full() {
            self.reserve(1);
            debug_assert!(!self.is_full());
        }

        let head = self.head;
        self.head = self.wrap_index(self.head + 1);
        unsafe { self.buffer_write(head, t) }
    }

    /// Deprecated: Renamed to `pop_back`.
    #[deprecated = "Renamed to `pop_back`"]
    pub fn pop(&mut self) -> Option<T> {
        self.pop_back()
    }

    /// Removes the last element from a buffer and returns it, or `None` if
    /// it is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::collections::RingBuf;
    ///
    /// let mut buf = RingBuf::new();
    /// assert_eq!(buf.pop_back(), None);
    /// buf.push_back(1i);
    /// buf.push_back(3);
    /// assert_eq!(buf.pop_back(), Some(3));
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            self.head = self.wrap_index(self.head - 1);
            let head = self.head;
            unsafe { Some(self.buffer_read(head)) }
        }
    }
}

/// Returns the index in the underlying buffer for a given logical element index.
#[inline]
fn wrap_index(index: uint, size: uint) -> uint {
    // size is always a power of 2
    index & (size - 1)
}

/// Calculate the number of elements left to be read in the buffer
#[inline]
fn count(tail: uint, head: uint, size: uint) -> uint {
    // size is always a power of 2
    (head - tail) & (size - 1)
}

/// `RingBuf` iterator.
pub struct Items<'a, T:'a> {
    ring: &'a [T],
    tail: uint,
    head: uint
}

impl<'a, T> Iterator<&'a T> for Items<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.tail == self.head {
            return None;
        }
        let tail = self.tail;
        self.tail = wrap_index(self.tail + 1, self.ring.len());
        unsafe { Some(self.ring.unsafe_get(tail)) }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let len = count(self.tail, self.head, self.ring.len());
        (len, Some(len))
    }
}

impl<'a, T> DoubleEndedIterator<&'a T> for Items<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        if self.tail == self.head {
            return None;
        }
        self.head = wrap_index(self.head - 1, self.ring.len());
        unsafe { Some(self.ring.unsafe_get(self.head)) }
    }
}


impl<'a, T> ExactSize<&'a T> for Items<'a, T> {}

impl<'a, T> RandomAccessIterator<&'a T> for Items<'a, T> {
    #[inline]
    fn indexable(&self) -> uint {
        let (len, _) = self.size_hint();
        len
    }

    #[inline]
    fn idx(&mut self, j: uint) -> Option<&'a T> {
        if j >= self.indexable() {
            None
        } else {
            let idx = wrap_index(self.tail + j, self.ring.len());
            unsafe { Some(self.ring.unsafe_get(idx)) }
        }
    }
}

// FIXME This was implemented differently from Items because of a problem
//       with returning the mutable reference. I couldn't find a way to
//       make the lifetime checker happy so, but there should be a way.
/// `RingBuf` mutable iterator.
pub struct MutItems<'a, T:'a> {
    ptr: *mut T,
    tail: uint,
    head: uint,
    cap: uint,
    marker: marker::ContravariantLifetime<'a>,
    marker2: marker::NoCopy
}

impl<'a, T> Iterator<&'a mut T> for MutItems<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        if self.tail == self.head {
            return None;
        }
        let tail = self.tail;
        self.tail = wrap_index(self.tail + 1, self.cap);
        if mem::size_of::<T>() != 0 {
            unsafe { Some(&mut *self.ptr.offset(tail as int)) }
        } else {
            // use a non-zero pointer
            Some(unsafe { mem::transmute(1u) })
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let len = count(self.tail, self.head, self.cap);
        (len, Some(len))
    }
}

impl<'a, T> DoubleEndedIterator<&'a mut T> for MutItems<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut T> {
        if self.tail == self.head {
            return None;
        }
        self.head = wrap_index(self.head - 1, self.cap);
        unsafe { Some(&mut *self.ptr.offset(self.head as int)) }
    }
}

impl<'a, T> ExactSize<&'a mut T> for MutItems<'a, T> {}

impl<A: PartialEq> PartialEq for RingBuf<A> {
    fn eq(&self, other: &RingBuf<A>) -> bool {
        self.len() == other.len() &&
            self.iter().zip(other.iter()).all(|(a, b)| a.eq(b))
    }
    fn ne(&self, other: &RingBuf<A>) -> bool {
        !self.eq(other)
    }
}

impl<A: Eq> Eq for RingBuf<A> {}

impl<A: PartialOrd> PartialOrd for RingBuf<A> {
    fn partial_cmp(&self, other: &RingBuf<A>) -> Option<Ordering> {
        iter::order::partial_cmp(self.iter(), other.iter())
    }
}

impl<A: Ord> Ord for RingBuf<A> {
    #[inline]
    fn cmp(&self, other: &RingBuf<A>) -> Ordering {
        iter::order::cmp(self.iter(), other.iter())
    }
}

impl<S: Writer, A: Hash<S>> Hash<S> for RingBuf<A> {
    fn hash(&self, state: &mut S) {
        self.len().hash(state);
        for elt in self.iter() {
            elt.hash(state);
        }
    }
}

impl<A> Index<uint, A> for RingBuf<A> {
    #[inline]
    fn index<'a>(&'a self, i: &uint) -> &'a A {
        self.get(*i).expect("Out of bounds access")
    }
}

impl<A> IndexMut<uint, A> for RingBuf<A> {
    #[inline]
    fn index_mut<'a>(&'a mut self, i: &uint) -> &'a mut A {
        self.get_mut(*i).expect("Out of bounds access")
    }
}

impl<A> FromIterator<A> for RingBuf<A> {
    fn from_iter<T: Iterator<A>>(iterator: T) -> RingBuf<A> {
        let (lower, _) = iterator.size_hint();
        let mut deq = RingBuf::with_capacity(lower);
        deq.extend(iterator);
        deq
    }
}

impl<A> Extend<A> for RingBuf<A> {
    fn extend<T: Iterator<A>>(&mut self, mut iterator: T) {
        for elt in iterator {
            self.push_back(elt);
        }
    }
}

impl<T: fmt::Show> fmt::Show for RingBuf<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "["));

        for (i, e) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}", *e));
        }

        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use self::Taggy::*;
    use self::Taggypar::*;
    use std::fmt::Show;
    use std::prelude::*;
    use std::hash;
    use test::Bencher;
    use test;

    use super::RingBuf;
    use vec::Vec;

    #[test]
    #[allow(deprecated)]
    fn test_simple() {
        let mut d = RingBuf::new();
        assert_eq!(d.len(), 0u);
        d.push_front(17i);
        d.push_front(42i);
        d.push_back(137);
        assert_eq!(d.len(), 3u);
        d.push_back(137);
        assert_eq!(d.len(), 4u);
        debug!("{}", d.front());
        assert_eq!(*d.front().unwrap(), 42);
        debug!("{}", d.back());
        assert_eq!(*d.back().unwrap(), 137);
        let mut i = d.pop_front();
        debug!("{}", i);
        assert_eq!(i, Some(42));
        i = d.pop_back();
        debug!("{}", i);
        assert_eq!(i, Some(137));
        i = d.pop_back();
        debug!("{}", i);
        assert_eq!(i, Some(137));
        i = d.pop_back();
        debug!("{}", i);
        assert_eq!(i, Some(17));
        assert_eq!(d.len(), 0u);
        d.push_back(3);
        assert_eq!(d.len(), 1u);
        d.push_front(2);
        assert_eq!(d.len(), 2u);
        d.push_back(4);
        assert_eq!(d.len(), 3u);
        d.push_front(1);
        assert_eq!(d.len(), 4u);
        debug!("{}", d[0]);
        debug!("{}", d[1]);
        debug!("{}", d[2]);
        debug!("{}", d[3]);
        assert_eq!(d[0], 1);
        assert_eq!(d[1], 2);
        assert_eq!(d[2], 3);
        assert_eq!(d[3], 4);
    }

    #[cfg(test)]
    fn test_parameterized<T:Clone + PartialEq + Show>(a: T, b: T, c: T, d: T) {
        let mut deq = RingBuf::new();
        assert_eq!(deq.len(), 0);
        deq.push_front(a.clone());
        deq.push_front(b.clone());
        deq.push_back(c.clone());
        assert_eq!(deq.len(), 3);
        deq.push_back(d.clone());
        assert_eq!(deq.len(), 4);
        assert_eq!((*deq.front().unwrap()).clone(), b.clone());
        assert_eq!((*deq.back().unwrap()).clone(), d.clone());
        assert_eq!(deq.pop_front().unwrap(), b.clone());
        assert_eq!(deq.pop_back().unwrap(), d.clone());
        assert_eq!(deq.pop_back().unwrap(), c.clone());
        assert_eq!(deq.pop_back().unwrap(), a.clone());
        assert_eq!(deq.len(), 0);
        deq.push_back(c.clone());
        assert_eq!(deq.len(), 1);
        deq.push_front(b.clone());
        assert_eq!(deq.len(), 2);
        deq.push_back(d.clone());
        assert_eq!(deq.len(), 3);
        deq.push_front(a.clone());
        assert_eq!(deq.len(), 4);
        assert_eq!(deq[0].clone(), a.clone());
        assert_eq!(deq[1].clone(), b.clone());
        assert_eq!(deq[2].clone(), c.clone());
        assert_eq!(deq[3].clone(), d.clone());
    }

    #[test]
    fn test_push_front_grow() {
        let mut deq = RingBuf::new();
        for i in range(0u, 66) {
            deq.push_front(i);
        }
        assert_eq!(deq.len(), 66);

        for i in range(0u, 66) {
            assert_eq!(deq[i], 65 - i);
        }

        let mut deq = RingBuf::new();
        for i in range(0u, 66) {
            deq.push_back(i);
        }

        for i in range(0u, 66) {
            assert_eq!(deq[i], i);
        }
    }

    #[test]
    fn test_index() {
        let mut deq = RingBuf::new();
        for i in range(1u, 4) {
            deq.push_front(i);
        }
        assert_eq!(deq[1], 2);
    }

    #[test]
    #[should_fail]
    fn test_index_out_of_bounds() {
        let mut deq = RingBuf::new();
        for i in range(1u, 4) {
            deq.push_front(i);
        }
        deq[3];
    }

    #[bench]
    fn bench_new(b: &mut test::Bencher) {
        b.iter(|| {
            let ring: RingBuf<u64> = RingBuf::new();
            test::black_box(ring);
        })
    }

    #[bench]
    fn bench_push_back_100(b: &mut test::Bencher) {
        let mut deq = RingBuf::with_capacity(101);
        b.iter(|| {
            for i in range(0i, 100) {
                deq.push_back(i);
            }
            deq.head = 0;
            deq.tail = 0;
        })
    }

    #[bench]
    fn bench_push_front_100(b: &mut test::Bencher) {
        let mut deq = RingBuf::with_capacity(101);
        b.iter(|| {
            for i in range(0i, 100) {
                deq.push_front(i);
            }
            deq.head = 0;
            deq.tail = 0;
        })
    }

    #[bench]
    fn bench_pop_back_100(b: &mut test::Bencher) {
        let mut deq: RingBuf<int> = RingBuf::with_capacity(101);

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
        let mut deq: RingBuf<int> = RingBuf::with_capacity(101);

        b.iter(|| {
            deq.head = 100;
            deq.tail = 0;
            while !deq.is_empty() {
                test::black_box(deq.pop_front());
            }
        })
    }

    #[bench]
    fn bench_grow_1025(b: &mut test::Bencher) {
        b.iter(|| {
            let mut deq = RingBuf::new();
            for i in range(0i, 1025) {
                deq.push_front(i);
            }
            test::black_box(deq);
        })
    }

    #[bench]
    fn bench_iter_1000(b: &mut test::Bencher) {
        let ring: RingBuf<int> = range(0i, 1000).collect();

        b.iter(|| {
            let mut sum = 0;
            for &i in ring.iter() {
                sum += i;
            }
            test::black_box(sum);
        })
    }

    #[bench]
    fn bench_mut_iter_1000(b: &mut test::Bencher) {
        let mut ring: RingBuf<int> = range(0i, 1000).collect();

        b.iter(|| {
            let mut sum = 0;
            for i in ring.iter_mut() {
                sum += *i;
            }
            test::black_box(sum);
        })
    }


    #[deriving(Clone, PartialEq, Show)]
    enum Taggy {
        One(int),
        Two(int, int),
        Three(int, int, int),
    }

    #[deriving(Clone, PartialEq, Show)]
    enum Taggypar<T> {
        Onepar(int),
        Twopar(int, int),
        Threepar(int, int, int),
    }

    #[deriving(Clone, PartialEq, Show)]
    struct RecCy {
        x: int,
        y: int,
        t: Taggy
    }

    #[test]
    fn test_param_int() {
        test_parameterized::<int>(5, 72, 64, 175);
    }

    #[test]
    fn test_param_taggy() {
        test_parameterized::<Taggy>(One(1), Two(1, 2), Three(1, 2, 3), Two(17, 42));
    }

    #[test]
    fn test_param_taggypar() {
        test_parameterized::<Taggypar<int>>(Onepar::<int>(1),
                                            Twopar::<int>(1, 2),
                                            Threepar::<int>(1, 2, 3),
                                            Twopar::<int>(17, 42));
    }

    #[test]
    fn test_param_reccy() {
        let reccy1 = RecCy { x: 1, y: 2, t: One(1) };
        let reccy2 = RecCy { x: 345, y: 2, t: Two(1, 2) };
        let reccy3 = RecCy { x: 1, y: 777, t: Three(1, 2, 3) };
        let reccy4 = RecCy { x: 19, y: 252, t: Two(17, 42) };
        test_parameterized::<RecCy>(reccy1, reccy2, reccy3, reccy4);
    }

    #[test]
    fn test_with_capacity() {
        let mut d = RingBuf::with_capacity(0);
        d.push_back(1i);
        assert_eq!(d.len(), 1);
        let mut d = RingBuf::with_capacity(50);
        d.push_back(1i);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_with_capacity_non_power_two() {
        let mut d3 = RingBuf::with_capacity(3);
        d3.push_back(1i);

        // X = None, | = lo
        // [|1, X, X]
        assert_eq!(d3.pop_front(), Some(1));
        // [X, |X, X]
        assert_eq!(d3.front(), None);

        // [X, |3, X]
        d3.push_back(3);
        // [X, |3, 6]
        d3.push_back(6);
        // [X, X, |6]
        assert_eq!(d3.pop_front(), Some(3));

        // Pushing the lo past half way point to trigger
        // the 'B' scenario for growth
        // [9, X, |6]
        d3.push_back(9);
        // [9, 12, |6]
        d3.push_back(12);

        d3.push_back(15);
        // There used to be a bug here about how the
        // RingBuf made growth assumptions about the
        // underlying Vec which didn't hold and lead
        // to corruption.
        // (Vec grows to next power of two)
        //good- [9, 12, 15, X, X, X, X, |6]
        //bug-  [15, 12, X, X, X, |6, X, X]
        assert_eq!(d3.pop_front(), Some(6));

        // Which leads us to the following state which
        // would be a failure case.
        //bug-  [15, 12, X, X, X, X, |X, X]
        assert_eq!(d3.front(), Some(&9));
    }

    #[test]
    fn test_reserve_exact() {
        let mut d = RingBuf::new();
        d.push_back(0u64);
        d.reserve_exact(50);
        assert!(d.capacity() >= 51);
        let mut d = RingBuf::new();
        d.push_back(0u32);
        d.reserve_exact(50);
        assert!(d.capacity() >= 51);
    }

    #[test]
    fn test_reserve() {
        let mut d = RingBuf::new();
        d.push_back(0u64);
        d.reserve(50);
        assert!(d.capacity() >= 51);
        let mut d = RingBuf::new();
        d.push_back(0u32);
        d.reserve(50);
        assert!(d.capacity() >= 51);
    }

    #[test]
    fn test_swap() {
        let mut d: RingBuf<int> = range(0i, 5).collect();
        d.pop_front();
        d.swap(0, 3);
        assert_eq!(d.iter().map(|&x|x).collect::<Vec<int>>(), vec!(4, 2, 3, 1));
    }

    #[test]
    fn test_iter() {
        let mut d = RingBuf::new();
        assert_eq!(d.iter().next(), None);
        assert_eq!(d.iter().size_hint(), (0, Some(0)));

        for i in range(0i, 5) {
            d.push_back(i);
        }
        {
            let b: &[_] = &[&0,&1,&2,&3,&4];
            assert_eq!(d.iter().collect::<Vec<&int>>().as_slice(), b);
        }

        for i in range(6i, 9) {
            d.push_front(i);
        }
        {
            let b: &[_] = &[&8,&7,&6,&0,&1,&2,&3,&4];
            assert_eq!(d.iter().collect::<Vec<&int>>().as_slice(), b);
        }

        let mut it = d.iter();
        let mut len = d.len();
        loop {
            match it.next() {
                None => break,
                _ => { len -= 1; assert_eq!(it.size_hint(), (len, Some(len))) }
            }
        }
    }

    #[test]
    fn test_rev_iter() {
        let mut d = RingBuf::new();
        assert_eq!(d.iter().rev().next(), None);

        for i in range(0i, 5) {
            d.push_back(i);
        }
        {
            let b: &[_] = &[&4,&3,&2,&1,&0];
            assert_eq!(d.iter().rev().collect::<Vec<&int>>().as_slice(), b);
        }

        for i in range(6i, 9) {
            d.push_front(i);
        }
        let b: &[_] = &[&4,&3,&2,&1,&0,&6,&7,&8];
        assert_eq!(d.iter().rev().collect::<Vec<&int>>().as_slice(), b);
    }

    #[test]
    fn test_mut_rev_iter_wrap() {
        let mut d = RingBuf::with_capacity(3);
        assert!(d.iter_mut().rev().next().is_none());

        d.push_back(1i);
        d.push_back(2);
        d.push_back(3);
        assert_eq!(d.pop_front(), Some(1));
        d.push_back(4);

        assert_eq!(d.iter_mut().rev().map(|x| *x).collect::<Vec<int>>(),
                   vec!(4, 3, 2));
    }

    #[test]
    fn test_mut_iter() {
        let mut d = RingBuf::new();
        assert!(d.iter_mut().next().is_none());

        for i in range(0u, 3) {
            d.push_front(i);
        }

        for (i, elt) in d.iter_mut().enumerate() {
            assert_eq!(*elt, 2 - i);
            *elt = i;
        }

        {
            let mut it = d.iter_mut();
            assert_eq!(*it.next().unwrap(), 0);
            assert_eq!(*it.next().unwrap(), 1);
            assert_eq!(*it.next().unwrap(), 2);
            assert!(it.next().is_none());
        }
    }

    #[test]
    fn test_mut_rev_iter() {
        let mut d = RingBuf::new();
        assert!(d.iter_mut().rev().next().is_none());

        for i in range(0u, 3) {
            d.push_front(i);
        }

        for (i, elt) in d.iter_mut().rev().enumerate() {
            assert_eq!(*elt, i);
            *elt = i;
        }

        {
            let mut it = d.iter_mut().rev();
            assert_eq!(*it.next().unwrap(), 0);
            assert_eq!(*it.next().unwrap(), 1);
            assert_eq!(*it.next().unwrap(), 2);
            assert!(it.next().is_none());
        }
    }

    #[test]
    fn test_from_iter() {
        use std::iter;
        let v = vec!(1i,2,3,4,5,6,7);
        let deq: RingBuf<int> = v.iter().map(|&x| x).collect();
        let u: Vec<int> = deq.iter().map(|&x| x).collect();
        assert_eq!(u, v);

        let mut seq = iter::count(0u, 2).take(256);
        let deq: RingBuf<uint> = seq.collect();
        for (i, &x) in deq.iter().enumerate() {
            assert_eq!(2*i, x);
        }
        assert_eq!(deq.len(), 256);
    }

    #[test]
    fn test_clone() {
        let mut d = RingBuf::new();
        d.push_front(17i);
        d.push_front(42);
        d.push_back(137);
        d.push_back(137);
        assert_eq!(d.len(), 4u);
        let mut e = d.clone();
        assert_eq!(e.len(), 4u);
        while !d.is_empty() {
            assert_eq!(d.pop_back(), e.pop_back());
        }
        assert_eq!(d.len(), 0u);
        assert_eq!(e.len(), 0u);
    }

    #[test]
    fn test_eq() {
        let mut d = RingBuf::new();
        assert!(d == RingBuf::with_capacity(0));
        d.push_front(137i);
        d.push_front(17);
        d.push_front(42);
        d.push_back(137);
        let mut e = RingBuf::with_capacity(0);
        e.push_back(42);
        e.push_back(17);
        e.push_back(137);
        e.push_back(137);
        assert!(&e == &d);
        e.pop_back();
        e.push_back(0);
        assert!(e != d);
        e.clear();
        assert!(e == RingBuf::new());
    }

    #[test]
    fn test_hash() {
      let mut x = RingBuf::new();
      let mut y = RingBuf::new();

      x.push_back(1i);
      x.push_back(2);
      x.push_back(3);

      y.push_back(0i);
      y.push_back(1i);
      y.pop_front();
      y.push_back(2);
      y.push_back(3);

      assert!(hash::hash(&x) == hash::hash(&y));
    }

    #[test]
    fn test_ord() {
        let x = RingBuf::new();
        let mut y = RingBuf::new();
        y.push_back(1i);
        y.push_back(2);
        y.push_back(3);
        assert!(x < y);
        assert!(y > x);
        assert!(x <= x);
        assert!(x >= x);
    }

    #[test]
    fn test_show() {
        let ringbuf: RingBuf<int> = range(0i, 10).collect();
        assert!(format!("{}", ringbuf).as_slice() == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");

        let ringbuf: RingBuf<&str> = vec!["just", "one", "test", "more"].iter()
                                                                        .map(|&s| s)
                                                                        .collect();
        assert!(format!("{}", ringbuf).as_slice() == "[just, one, test, more]");
    }

    #[test]
    fn test_drop() {
        static mut drops: uint = 0;
        struct Elem;
        impl Drop for Elem {
            fn drop(&mut self) {
                unsafe { drops += 1; }
            }
        }

        let mut ring = RingBuf::new();
        ring.push_back(Elem);
        ring.push_front(Elem);
        ring.push_back(Elem);
        ring.push_front(Elem);
        drop(ring);

        assert_eq!(unsafe {drops}, 4);
    }

    #[test]
    fn test_drop_with_pop() {
        static mut drops: uint = 0;
        struct Elem;
        impl Drop for Elem {
            fn drop(&mut self) {
                unsafe { drops += 1; }
            }
        }

        let mut ring = RingBuf::new();
        ring.push_back(Elem);
        ring.push_front(Elem);
        ring.push_back(Elem);
        ring.push_front(Elem);

        drop(ring.pop_back());
        drop(ring.pop_front());
        assert_eq!(unsafe {drops}, 2);

        drop(ring);
        assert_eq!(unsafe {drops}, 4);
    }

    #[test]
    fn test_drop_clear() {
        static mut drops: uint = 0;
        struct Elem;
        impl Drop for Elem {
            fn drop(&mut self) {
                unsafe { drops += 1; }
            }
        }

        let mut ring = RingBuf::new();
        ring.push_back(Elem);
        ring.push_front(Elem);
        ring.push_back(Elem);
        ring.push_front(Elem);
        ring.clear();
        assert_eq!(unsafe {drops}, 4);

        drop(ring);
        assert_eq!(unsafe {drops}, 4);
    }

    #[test]
    fn test_reserve_grow() {
        // test growth path A
        // [T o o H] -> [T o o H . . . . ]
        let mut ring = RingBuf::with_capacity(4);
        for i in range(0i, 3) {
            ring.push_back(i);
        }
        ring.reserve(7);
        for i in range(0i, 3) {
            assert_eq!(ring.pop_front(), Some(i));
        }

        // test growth path B
        // [H T o o] -> [. T o o H . . . ]
        let mut ring = RingBuf::with_capacity(4);
        for i in range(0i, 1) {
            ring.push_back(i);
            assert_eq!(ring.pop_front(), Some(i));
        }
        for i in range(0i, 3) {
            ring.push_back(i);
        }
        ring.reserve(7);
        for i in range(0i, 3) {
            assert_eq!(ring.pop_front(), Some(i));
        }

        // test growth path C
        // [o o H T] -> [o o H . . . . T ]
        let mut ring = RingBuf::with_capacity(4);
        for i in range(0i, 3) {
            ring.push_back(i);
            assert_eq!(ring.pop_front(), Some(i));
        }
        for i in range(0i, 3) {
            ring.push_back(i);
        }
        ring.reserve(7);
        for i in range(0i, 3) {
            assert_eq!(ring.pop_front(), Some(i));
        }
    }

    #[test]
    fn test_get() {
        let mut ring = RingBuf::new();
        ring.push_back(0i);
        assert_eq!(ring.get(0), Some(&0));
        assert_eq!(ring.get(1), None);

        ring.push_back(1);
        assert_eq!(ring.get(0), Some(&0));
        assert_eq!(ring.get(1), Some(&1));
        assert_eq!(ring.get(2), None);

        ring.push_back(2);
        assert_eq!(ring.get(0), Some(&0));
        assert_eq!(ring.get(1), Some(&1));
        assert_eq!(ring.get(2), Some(&2));
        assert_eq!(ring.get(3), None);

        assert_eq!(ring.pop_front(), Some(0));
        assert_eq!(ring.get(0), Some(&1));
        assert_eq!(ring.get(1), Some(&2));
        assert_eq!(ring.get(2), None);

        assert_eq!(ring.pop_front(), Some(1));
        assert_eq!(ring.get(0), Some(&2));
        assert_eq!(ring.get(1), None);

        assert_eq!(ring.pop_front(), Some(2));
        assert_eq!(ring.get(0), None);
        assert_eq!(ring.get(1), None);
    }

    #[test]
    fn test_get_mut() {
        let mut ring = RingBuf::new();
        for i in range(0i, 3) {
            ring.push_back(i);
        }

        match ring.get_mut(1) {
            Some(x) => *x = -1,
            None => ()
        };

        assert_eq!(ring.get_mut(0), Some(&mut 0));
        assert_eq!(ring.get_mut(1), Some(&mut -1));
        assert_eq!(ring.get_mut(2), Some(&mut 2));
        assert_eq!(ring.get_mut(3), None);

        assert_eq!(ring.pop_front(), Some(0));
        assert_eq!(ring.get_mut(0), Some(&mut -1));
        assert_eq!(ring.get_mut(1), Some(&mut 2));
        assert_eq!(ring.get_mut(2), None);
    }
}
