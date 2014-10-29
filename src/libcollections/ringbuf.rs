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
use core::num;

use std::hash::{Writer, Hash};
use std::cmp;

use alloc::heap;

use {Deque, Mutable, MutableSeq};

static INITIAL_CAPACITY: uint = 8u; // 2^3
static MINIMUM_CAPACITY: uint = 2u;

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
        while self.len() > 0 {
            drop(self.pop_front())
        }
        unsafe {
            heap::deallocate(self.ptr as *mut u8,
                             self.cap * mem::size_of::<T>(),
                             mem::min_align_of::<T>())
        }
    }
}

impl<T> Collection for RingBuf<T> {
    /// Returns the number of elements in the `RingBuf`.
    fn len(&self) -> uint { count(self.tail, self.head, self.cap) }
}

impl<T> Mutable for RingBuf<T> {
    /// Clears the `RingBuf`, removing all values.
    fn clear(&mut self) {
        while self.tail != self.head {
            let _ = self.pop_front();
        }
        self.tail = 0;
        self.head = 0;
    }
}

impl<T> Deque<T> for RingBuf<T> {
    /// Returns a reference to the first element in the `RingBuf`.
    fn front<'a>(&'a self) -> Option<&'a T> { self.get(0) }

    /// Returns a mutable reference to the first element in the `RingBuf`.
    fn front_mut<'a>(&'a mut self) -> Option<&'a mut T> { self.get_mut(0) }

    /// Returns a reference to the last element in the `RingBuf`.
    fn back<'a>(&'a self) -> Option<&'a T> {
        let idx = self.len() - 1;
        self.get(idx)
    }

    /// Returns a mutable reference to the last element in the `RingBuf`.
    fn back_mut<'a>(&'a mut self) -> Option<&'a mut T> {
        let idx = self.len() - 1;
        self.get_mut(idx)
    }

    /// Removes and returns the first element in the `RingBuf`, or `None` if it
    /// is empty.
    fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let tail = self.tail;
            self.tail = wrap_index(self.tail + 1, self.cap);
            unsafe { Some(self.buffer_read(tail)) }
        }
    }

    /// Prepends an element to the `RingBuf`.
    fn push_front(&mut self, t: T) {
        if self.is_full() {
            let newcap = self.cap * 2;
            self.grow(newcap);
        }

        self.tail = wrap_index(self.tail - 1, self.cap);
        let tail = self.tail;
        unsafe { self.buffer_write(tail, t); }
    }
}

impl<T> MutableSeq<T> for RingBuf<T> {
    fn push(&mut self, t: T) {
        if self.is_full() {
            let newcap = self.cap * 2;
            self.grow(newcap);
        }

        let head = self.head;
        self.head = wrap_index(self.head + 1, self.cap);
        unsafe { self.buffer_write(head, t) }
    }

    fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            self.head = wrap_index(self.head - 1, self.cap);
            let head = self.head;
            unsafe { Some(self.buffer_read(head)) }
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
    fn buffer_as_slice(&self) -> &[T] {
        unsafe { mem::transmute(RawSlice { data: self.ptr as *const T, len: self.cap }) }
    }

    /// Turn ptr into a mutable slice
    #[inline]
    fn buffer_as_slice_mut(&mut self) -> &mut [T] {
        unsafe { mem::transmute(RawSlice { data: self.ptr as *const T, len: self.cap }) }
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

    /// Grows the buffer to a new length, this expected the new size
    /// to be greated then the current size and a power of two.
    #[inline]
    fn grow(&mut self, newlen: uint) {
        assert!(newlen > self.cap);
        assert!(newlen == num::next_power_of_two(newlen));
        let old = self.cap * mem::size_of::<T>();
        let new = newlen.checked_mul(&mem::size_of::<T>())
                        .expect("capacity overflow");
        unsafe {
            self.ptr = heap::reallocate(self.ptr as *mut u8,
                                        old,
                                        new,
                                        mem::min_align_of::<T>()) as *mut T;
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

        let oldlen = self.cap;
        self.cap = newlen;

        if self.tail <= self.head { // A
            // Nop
        } else if self.head < oldlen - self.tail { // B
            unsafe {
                ptr::copy_nonoverlapping_memory(
                    self.ptr.offset(oldlen as int) as *mut T,
                    self.ptr.offset(0 as int) as *const T,
                    self.head
                );
            }
            self.head += oldlen;
        } else { // C
            unsafe {
                ptr::copy_nonoverlapping_memory(
                    self.ptr.offset((newlen - oldlen + self.tail) as int) as *mut T,
                    self.ptr.offset(self.tail as int) as *const T,
                    oldlen - self.tail
                );
            }
            self.tail = newlen - oldlen + self.tail;
        }
    }
}

impl<T> RingBuf<T> {
    /// Creates an empty `RingBuf`.
    pub fn new() -> RingBuf<T> {
        RingBuf::with_capacity(INITIAL_CAPACITY)
    }

    /// Creates an empty `RingBuf` with space for at least `n` elements.
    pub fn with_capacity(n: uint) -> RingBuf<T> {
        // +1 since the ringbuffer always leaves one space empty
        let cap = num::next_power_of_two(cmp::max(n + 1, MINIMUM_CAPACITY));
        let size = cap.checked_mul(&mem::size_of::<T>())
                      .expect("capacity overflow");

        RingBuf {
            tail: 0,
            head: 0,
            cap: cap,
            ptr: unsafe { heap::allocate(size, mem::min_align_of::<T>()) as *mut T }
        }
    }

    /// Retrieves an element in the `RingBuf` by index.
    ///
    /// Returns None if there is no element with the given index.
    ///
    /// # Example
    ///
    /// ```rust
    /// #![allow(deprecated)]
    ///
    /// use std::collections::RingBuf;
    ///
    /// let mut buf = RingBuf::new();
    /// buf.push(3i);
    /// buf.push(4);
    /// buf.push(5);
    /// assert_eq!(buf.get(1), Some(&4));
    /// ```
    pub fn get<'a>(&'a self, i: uint) -> Option<&'a T> {
        if self.len() > i {
            let idx = wrap_index(self.tail + i, self.cap);
            unsafe { Some(self.buffer_as_slice().unsafe_get(idx)) }
        } else {
            None
        }
    }

    /// Retrieves an element in the `RingBuf` by index.
    ///
    /// Returns None if there is no element with the given index.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::collections::RingBuf;
    ///
    /// let mut buf = RingBuf::new();
    /// buf.push(3i);
    /// buf.push(4);
    /// buf.push(5);
    /// match buf.get_mut(1) {
    ///     Some(v) => *v = 7,
    ///     None => ()
    /// };
    /// assert_eq!(buf[1], 7);
    /// ```
    pub fn get_mut<'a>(&'a mut self, i: uint) -> Option<&'a mut T> {
        if self.len() > i {
            let idx = wrap_index(self.tail + i, self.cap);
            unsafe { Some(self.buffer_as_slice_mut().unsafe_mut(idx)) }
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
    /// buf.push(3i);
    /// buf.push(4);
    /// buf.push(5);
    /// buf.swap(0, 2);
    /// assert_eq!(buf[0], 5);
    /// assert_eq!(buf[2], 3);
    /// ```
    pub fn swap(&mut self, i: uint, j: uint) {
        assert!(i < self.len());
        assert!(j < self.len());
        let ri = wrap_index(self.tail + i, self.cap);
        let rj = wrap_index(self.tail + j, self.cap);
        self.buffer_as_slice_mut().swap(ri, rj);
    }

    /// Reserves capacity for exactly `n` elements in the given `RingBuf`,
    /// doing nothing if `self`'s capacity is already equal to or greater
    /// than the requested capacity.
    #[deprecated = "use reserve, Ringbuf can no longer be an exact size."]
    pub fn reserve_exact(&mut self, n: uint) {
        self.reserve(n);
    }

    /// Reserves capacity for at least `n` elements in the given `RingBuf`,
    /// over-allocating in case the caller needs to reserve additional
    /// space.
    ///
    /// Do nothing if `self`'s capacity is already equal to or greater
    /// than the requested capacity.
    pub fn reserve(&mut self, n: uint) {
        // +1 since the buffer needs one more space then the expected size.
        let count = num::next_power_of_two(n + 1);
        if count > self.cap {
            self.grow(count);
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
    /// buf.push(5i);
    /// buf.push(3);
    /// buf.push(4);
    /// let b: &[_] = &[&5, &3, &4];
    /// assert_eq!(buf.iter().collect::<Vec<&int>>().as_slice(), b);
    /// ```
    pub fn iter<'a>(&'a self) -> Items<'a, T> {
        Items {
            tail: self.tail,
            head: self.head,
            ring: self.buffer_as_slice()
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
    /// buf.push(5i);
    /// buf.push(3);
    /// buf.push(4);
    /// for num in buf.iter_mut() {
    ///     *num = *num - 2;
    /// }
    /// let b: &[_] = &[&mut 3, &mut 1, &mut 2];
    /// assert_eq!(buf.iter_mut().collect::<Vec<&mut int>>()[], b);
    /// ```
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


/// `RingBuf` iterator.
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
        unsafe { Some(&mut *self.ptr.offset(tail as int)) }
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
        self.get(*i).expect("Index out of bounds")
    }
}

impl<A> IndexMut<uint, A> for RingBuf<A> {
    #[inline]
    fn index_mut<'a>(&'a mut self, index: &uint) -> &'a mut A {
        self.get_mut(*index).expect("Index out of bounds")
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

impl<A> Extendable<A> for RingBuf<A> {
    fn extend<T: Iterator<A>>(&mut self, mut iterator: T) {
        for elt in iterator {
            self.push(elt);
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
    use std::fmt::Show;
    use std::prelude::*;
    use std::hash;
    use test::Bencher;
    use test;

    use {Deque, Mutable, MutableSeq};
    use super::RingBuf;
    use vec::Vec;

    #[test]
    #[allow(deprecated)]
    fn test_simple() {
        let mut d = RingBuf::new();
        assert_eq!(d.len(), 0u);
        d.push_front(17i);
        d.push_front(42i);
        d.push(137);
        assert_eq!(d.len(), 3u);
        d.push(137);
        assert_eq!(d.len(), 4u);
        debug!("{}", d.front());
        assert_eq!(*d.front().unwrap(), 42);
        debug!("{}", d.back());
        assert_eq!(*d.back().unwrap(), 137);
        let mut i = d.pop_front();
        debug!("{}", i);
        assert_eq!(i, Some(42));
        i = d.pop();
        debug!("{}", i);
        assert_eq!(i, Some(137));
        i = d.pop();
        debug!("{}", i);
        assert_eq!(i, Some(137));
        i = d.pop();
        debug!("{}", i);
        assert_eq!(i, Some(17));
        assert_eq!(d.len(), 0u);
        d.push(3);
        assert_eq!(d.len(), 1u);
        d.push_front(2);
        assert_eq!(d.len(), 2u);
        d.push(4);
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
        deq.push(c.clone());
        assert_eq!(deq.len(), 3);
        deq.push(d.clone());
        assert_eq!(deq.len(), 4);
        assert_eq!((*deq.front().unwrap()).clone(), b.clone());
        assert_eq!((*deq.back().unwrap()).clone(), d.clone());
        assert_eq!(deq.pop_front().unwrap(), b.clone());
        assert_eq!(deq.pop().unwrap(), d.clone());
        assert_eq!(deq.pop().unwrap(), c.clone());
        assert_eq!(deq.pop().unwrap(), a.clone());
        assert_eq!(deq.len(), 0);
        deq.push(c.clone());
        assert_eq!(deq.len(), 1);
        deq.push_front(b.clone());
        assert_eq!(deq.len(), 2);
        deq.push(d.clone());
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
            deq.push(i);
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
            let _: RingBuf<u64> = RingBuf::new();
        })
    }

    #[bench]
    fn bench_push_back_100(b: &mut test::Bencher) {
        let mut deq = RingBuf::with_capacity(100);
        b.iter(|| {
            for i in range(0i, 100) {
                deq.push(i);
            }
            deq.clear();
        })
    }

    #[bench]
    fn bench_push_front_100(b: &mut test::Bencher) {
        let mut deq = RingBuf::with_capacity(100);
        b.iter(|| {
            for i in range(0i, 100) {
                deq.push_front(i);
            }
            deq.clear();
        })
    }

    #[bench]
    fn bench_pop_100(b: &mut test::Bencher) {
        let mut deq = RingBuf::with_capacity(100);

        b.iter(|| {
            for i in range(0i, 100) {
                deq.push(i);
            }
            while None != deq.pop() {}
        })
    }

    #[bench]
    fn bench_pop_front_100(b: &mut test::Bencher) {
        let mut deq = RingBuf::with_capacity(100);

        b.iter(|| {
            for i in range(0i, 100) {
                deq.push(i);
            }
            while None != deq.pop_front() {}
        })
    }

    #[bench]
    fn bench_grow_1025(b: &mut test::Bencher) {
        b.iter(|| {
            let mut deq = RingBuf::new();
            for i in range(0i, 1025) {
                deq.push_front(i);
            }
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
            sum
        })
    }

    #[bench]
    fn bench_mut_iter_1000(b: &mut test::Bencher) {
        let mut ring: RingBuf<int> = range(0i, 1000).collect();

        b.iter(|| {
            for i in ring.iter_mut() {
                *i += 1;
            }
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
        d.push(1i);
        assert_eq!(d.len(), 1);
        let mut d = RingBuf::with_capacity(50);
        d.push(1i);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_with_capacity_non_power_two() {
        let mut d3 = RingBuf::with_capacity(3);
        d3.push(1i);

        // X = None, | = lo
        // [|1, X, X]
        assert_eq!(d3.pop_front(), Some(1));
        // [X, |X, X]
        assert_eq!(d3.front(), None);

        // [X, |3, X]
        d3.push(3);
        // [X, |3, 6]
        d3.push(6);
        // [X, X, |6]
        assert_eq!(d3.pop_front(), Some(3));

        // Pushing the lo past half way point to trigger
        // the 'B' scenario for growth
        // [9, X, |6]
        d3.push(9);
        // [9, 12, |6]
        d3.push(12);

        d3.push(15);
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
    fn test_reserve() {
        let mut d = RingBuf::new();
        d.push(0u64);
        d.reserve(50);
        assert_eq!(d.cap, 64);
        let mut d = RingBuf::new();
        d.push(0u32);
        d.reserve(50);
        assert_eq!(d.cap, 64);
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
            d.push(i);
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
            d.push(i);
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

        d.push(1i);
        d.push(2);
        d.push(3);
        assert_eq!(d.pop_front(), Some(1));
        d.push(4);

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
        d.push(137);
        d.push(137);
        assert_eq!(d.len(), 4u);
        let mut e = d.clone();
        assert_eq!(e.len(), 4u);
        while !d.is_empty() {
            assert_eq!(d.pop(), e.pop());
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
        d.push(137);
        let mut e = RingBuf::with_capacity(0);
        e.push(42);
        e.push(17);
        e.push(137);
        e.push(137);
        assert!(&e == &d);
        e.pop();
        e.push(0);
        assert!(e != d);
        e.clear();
        assert!(e == RingBuf::new());
    }

    #[test]
    fn test_hash() {
      let mut x = RingBuf::new();
      let mut y = RingBuf::new();

      x.push(1i);
      x.push(2);
      x.push(3);

      y.push(0i);
      y.push(1i);
      y.pop_front();
      y.push(2);
      y.push(3);

      assert!(hash::hash(&x) == hash::hash(&y));
    }

    #[test]
    fn test_ord() {
        let x = RingBuf::new();
        let mut y = RingBuf::new();
        y.push(1i);
        y.push(2);
        y.push(3);
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
        ring.push(Elem);
        ring.push_front(Elem);
        ring.push(Elem);
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
        ring.push(Elem);
        ring.push_front(Elem);
        ring.push(Elem);
        ring.push_front(Elem);

        drop(ring.pop());
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
        ring.push(Elem);
        ring.push_front(Elem);
        ring.push(Elem);
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
            ring.push(i);
        }
        ring.reserve(7);
        for i in range(0i, 3) {
            assert_eq!(ring.pop_front(), Some(i));
        }

        // test growth path B
        // [H T o o] -> [. T o o H . . . ]
        let mut ring = RingBuf::with_capacity(4);
        for i in range(0i, 1) {
            ring.push(i);
            assert_eq!(ring.pop_front(), Some(i));
        }
        for i in range(0i, 3) {
            ring.push(i);
        }
        ring.reserve(7);
        for i in range(0i, 3) {
            assert_eq!(ring.pop_front(), Some(i));
        }

        // test growth path C
        // [o o H T] -> [o o H . . . . T ]
        let mut ring = RingBuf::with_capacity(4);
        for i in range(0i, 3) {
            ring.push(i);
            assert_eq!(ring.pop_front(), Some(i));
        }
        for i in range(0i, 3) {
            ring.push(i);
        }
        ring.reserve(7);
        for i in range(0i, 3) {
            assert_eq!(ring.pop_front(), Some(i));
        }
    }
}
