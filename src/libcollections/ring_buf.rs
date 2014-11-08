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

use core::cmp;
use core::default::Default;
use core::fmt;
use core::iter;
use core::slice;
use std::hash::{Writer, Hash};

use vec::Vec;

static INITIAL_CAPACITY: uint = 8u; // 2^3
static MINIMUM_CAPACITY: uint = 2u;

// FIXME(conventions): implement shrink_to_fit. Awkward with the current design, but it should
// be scrapped anyway. Defer to rewrite?
// FIXME(conventions): implement into_iter


/// `RingBuf` is a circular buffer that implements `Deque`.
#[deriving(Clone)]
pub struct RingBuf<T> {
    nelts: uint,
    lo: uint,
    elts: Vec<Option<T>>
}

impl<T> Default for RingBuf<T> {
    #[inline]
    fn default() -> RingBuf<T> { RingBuf::new() }
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
        RingBuf{nelts: 0, lo: 0,
              elts: Vec::from_fn(cmp::max(MINIMUM_CAPACITY, n), |_| None)}
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
        match self.elts.get(i) {
            None => None,
            Some(opt) => opt.as_ref(),
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
        match self.elts.get_mut(i) {
            None => None,
            Some(opt) => opt.as_mut(),
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
        let ri = self.raw_index(i);
        let rj = self.raw_index(j);
        self.elts.as_mut_slice().swap(ri, rj);
    }

    /// Returns the index in the underlying `Vec` for a given logical element
    /// index.
    fn raw_index(&self, idx: uint) -> uint {
        raw_index(self.lo, self.elts.len(), idx)
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
    /// assert_eq!(buf.capacity(), 10);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn capacity(&self) -> uint {
        // FXIME(Gankro): not the actual usable capacity if you use reserve/reserve_exact
        self.elts.capacity()
    }

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
        // FIXME(Gankro): this is just wrong. The ringbuf won't actually use this space
        self.elts.reserve_exact(additional);
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
        // FIXME(Gankro): this is just wrong. The ringbuf won't actually use this space
        self.elts.reserve(additional);
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
        Items{index: 0, rindex: self.nelts, lo: self.lo, elts: self.elts.as_slice()}
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
    pub fn iter_mut(&mut self) -> MutItems<T> {
        let start_index = raw_index(self.lo, self.elts.len(), 0);
        let end_index = raw_index(self.lo, self.elts.len(), self.nelts);

        // Divide up the array
        if end_index <= start_index {
            // Items to iterate goes from:
            //    start_index to self.elts.len()
            // and then
            //    0 to end_index
            let (temp, remaining1) = self.elts.split_at_mut(start_index);
            let (remaining2, _) = temp.split_at_mut(end_index);
            MutItems {
                remaining1: remaining1.iter_mut(),
                remaining2: remaining2.iter_mut(),
                nelts: self.nelts,
            }
        } else {
            // Items to iterate goes from start_index to end_index:
            let (empty, elts) = self.elts.split_at_mut(0);
            let remaining1 = elts[mut start_index..end_index];
            MutItems {
                remaining1: remaining1.iter_mut(),
                remaining2: empty.iter_mut(),
                nelts: self.nelts,
            }
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
    pub fn len(&self) -> uint { self.nelts }

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
        for x in self.elts.iter_mut() { *x = None }
        self.nelts = 0;
        self.lo = 0;
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
        if self.nelts > 0 { Some(&self[0]) } else { None }
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
        if self.nelts > 0 { Some(&mut self[0]) } else { None }
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
        if self.nelts > 0 { Some(&self[self.nelts - 1]) } else { None }
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
        let nelts = self.nelts;
        if nelts > 0 { Some(&mut self[nelts - 1]) } else { None }
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
        let result = self.elts[self.lo].take();
        if result.is_some() {
            self.lo = (self.lo + 1u) % self.elts.len();
            self.nelts -= 1u;
        }
        result
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
        if self.nelts == self.elts.len() {
            grow(self.nelts, &mut self.lo, &mut self.elts);
        }
        if self.lo == 0u {
            self.lo = self.elts.len() - 1u;
        } else { self.lo -= 1u; }
        self.elts[self.lo] = Some(t);
        self.nelts += 1u;
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
        if self.nelts == self.elts.len() {
            grow(self.nelts, &mut self.lo, &mut self.elts);
        }
        let hi = self.raw_index(self.nelts);
        self.elts[hi] = Some(t);
        self.nelts += 1u;
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
        if self.nelts > 0 {
            self.nelts -= 1;
            let hi = self.raw_index(self.nelts);
            self.elts[hi].take()
        } else {
            None
        }
    }
}

/// `RingBuf` iterator.
pub struct Items<'a, T:'a> {
    lo: uint,
    index: uint,
    rindex: uint,
    elts: &'a [Option<T>],
}

impl<'a, T> Iterator<&'a T> for Items<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.index == self.rindex {
            return None;
        }
        let raw_index = raw_index(self.lo, self.elts.len(), self.index);
        self.index += 1;
        Some(self.elts[raw_index].as_ref().unwrap())
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let len = self.rindex - self.index;
        (len, Some(len))
    }
}

impl<'a, T> DoubleEndedIterator<&'a T> for Items<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        if self.index == self.rindex {
            return None;
        }
        self.rindex -= 1;
        let raw_index = raw_index(self.lo, self.elts.len(), self.rindex);
        Some(self.elts[raw_index].as_ref().unwrap())
    }
}

impl<'a, T> ExactSize<&'a T> for Items<'a, T> {}

impl<'a, T> RandomAccessIterator<&'a T> for Items<'a, T> {
    #[inline]
    fn indexable(&self) -> uint { self.rindex - self.index }

    #[inline]
    fn idx(&mut self, j: uint) -> Option<&'a T> {
        if j >= self.indexable() {
            None
        } else {
            let raw_index = raw_index(self.lo, self.elts.len(), self.index + j);
            Some(self.elts[raw_index].as_ref().unwrap())
        }
    }
}

/// `RingBuf` mutable iterator.
pub struct MutItems<'a, T:'a> {
    remaining1: slice::MutItems<'a, Option<T>>,
    remaining2: slice::MutItems<'a, Option<T>>,
    nelts: uint,
}

impl<'a, T> Iterator<&'a mut T> for MutItems<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        if self.nelts == 0 {
            return None;
        }
        self.nelts -= 1;
        match self.remaining1.next() {
            Some(ptr) => return Some(ptr.as_mut().unwrap()),
            None => {}
        }
        match self.remaining2.next() {
            Some(ptr) => return Some(ptr.as_mut().unwrap()),
            None => unreachable!(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.nelts, Some(self.nelts))
    }
}

impl<'a, T> DoubleEndedIterator<&'a mut T> for MutItems<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut T> {
        if self.nelts == 0 {
            return None;
        }
        self.nelts -= 1;
        match self.remaining2.next_back() {
            Some(ptr) => return Some(ptr.as_mut().unwrap()),
            None => {}
        }
        match self.remaining1.next_back() {
            Some(ptr) => return Some(ptr.as_mut().unwrap()),
            None => unreachable!(),
        }
    }
}

impl<'a, T> ExactSize<&'a mut T> for MutItems<'a, T> {}

/// Grow is only called on full elts, so nelts is also len(elts), unlike
/// elsewhere.
fn grow<T>(nelts: uint, loptr: &mut uint, elts: &mut Vec<Option<T>>) {
    assert_eq!(nelts, elts.len());
    let lo = *loptr;
    elts.reserve_exact(nelts);
    let newlen = elts.capacity();

    /* fill with None */
    for _ in range(elts.len(), newlen) {
        elts.push(None);
    }

    /*
      Move the shortest half into the newly reserved area.
      lo ---->|
      nelts ----------->|
        [o o o|o o o o o]
      A [. . .|o o o o o o o o|. . . . .]
      B [o o o|. . . . . . . .|o o o o o]
     */

    assert!(newlen - nelts/2 >= nelts);
    if lo <= (nelts - lo) { // A
        for i in range(0u, lo) {
            elts.as_mut_slice().swap(i, nelts + i);
        }
    } else {                // B
        for i in range(lo, nelts) {
            elts.as_mut_slice().swap(i, newlen - nelts + i);
        }
        *loptr += newlen - nelts;
    }
}

/// Returns the index in the underlying `Vec` for a given logical element index.
fn raw_index(lo: uint, len: uint, index: uint) -> uint {
    if lo >= len - index {
        lo + index - len
    } else {
        lo + index
    }
}

impl<A: PartialEq> PartialEq for RingBuf<A> {
    fn eq(&self, other: &RingBuf<A>) -> bool {
        self.nelts == other.nelts &&
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
        let idx = self.raw_index(*i);
        match self.elts[idx] {
            None => panic!(),
            Some(ref v) => v,
        }
    }
}

impl<A> IndexMut<uint, A> for RingBuf<A> {
    #[inline]
    fn index_mut<'a>(&'a mut self, i: &uint) -> &'a mut A {
        let idx = self.raw_index(*i);
        match *(&mut self.elts[idx]) {
            None => panic!(),
            Some(ref mut v) => v
        }
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
            let _: RingBuf<u64> = RingBuf::new();
        })
    }

    #[bench]
    fn bench_push_back(b: &mut test::Bencher) {
        let mut deq = RingBuf::new();
        b.iter(|| {
            deq.push_back(0i);
        })
    }

    #[bench]
    fn bench_push_front(b: &mut test::Bencher) {
        let mut deq = RingBuf::new();
        b.iter(|| {
            deq.push_front(0i);
        })
    }

    #[bench]
    fn bench_grow(b: &mut test::Bencher) {
        let mut deq = RingBuf::new();
        b.iter(|| {
            for _ in range(0i, 65) {
                deq.push_front(1i);
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
        assert!(d.capacity() >= 64);
        let mut d = RingBuf::new();
        d.push_back(0u32);
        d.reserve(50);
        assert!(d.capacity() >= 64);
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
}
