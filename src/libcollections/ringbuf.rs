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

use {Deque, Mutable, MutableSeq};
use vec::Vec;

static INITIAL_CAPACITY: uint = 8u; // 2^3
static MINIMUM_CAPACITY: uint = 2u;

/// `RingBuf` is a circular buffer that implements `Deque`.
#[deriving(Clone)]
pub struct RingBuf<T> {
    nelts: uint,
    lo: uint,
    elts: Vec<Option<T>>
}

impl<T> Collection for RingBuf<T> {
    /// Returns the number of elements in the `RingBuf`.
    fn len(&self) -> uint { self.nelts }
}

impl<T> Mutable for RingBuf<T> {
    /// Clears the `RingBuf`, removing all values.
    fn clear(&mut self) {
        for x in self.elts.iter_mut() { *x = None }
        self.nelts = 0;
        self.lo = 0;
    }
}

impl<T> Deque<T> for RingBuf<T> {
    /// Returns a reference to the first element in the `RingBuf`.
    fn front<'a>(&'a self) -> Option<&'a T> {
        if self.nelts > 0 { Some(&self[0]) } else { None }
    }

    /// Returns a mutable reference to the first element in the `RingBuf`.
    fn front_mut<'a>(&'a mut self) -> Option<&'a mut T> {
        if self.nelts > 0 { Some(self.get_mut(0)) } else { None }
    }

    /// Returns a reference to the last element in the `RingBuf`.
    fn back<'a>(&'a self) -> Option<&'a T> {
        if self.nelts > 0 { Some(&self[self.nelts - 1]) } else { None }
    }

    /// Returns a mutable reference to the last element in the `RingBuf`.
    fn back_mut<'a>(&'a mut self) -> Option<&'a mut T> {
        let nelts = self.nelts;
        if nelts > 0 { Some(self.get_mut(nelts - 1)) } else { None }
    }

    /// Removes and returns the first element in the `RingBuf`, or `None` if it
    /// is empty.
    fn pop_front(&mut self) -> Option<T> {
        let result = self.elts.get_mut(self.lo).take();
        if result.is_some() {
            self.lo = (self.lo + 1u) % self.elts.len();
            self.nelts -= 1u;
        }
        result
    }

    /// Prepends an element to the `RingBuf`.
    fn push_front(&mut self, t: T) {
        if self.nelts == self.elts.len() {
            grow(self.nelts, &mut self.lo, &mut self.elts);
        }
        if self.lo == 0u {
            self.lo = self.elts.len() - 1u;
        } else { self.lo -= 1u; }
        *self.elts.get_mut(self.lo) = Some(t);
        self.nelts += 1u;
    }
}

impl<T> MutableSeq<T> for RingBuf<T> {
    fn push(&mut self, t: T) {
        if self.nelts == self.elts.len() {
            grow(self.nelts, &mut self.lo, &mut self.elts);
        }
        let hi = self.raw_index(self.nelts);
        *self.elts.get_mut(hi) = Some(t);
        self.nelts += 1u;
    }
    fn pop(&mut self) -> Option<T> {
        if self.nelts > 0 {
            self.nelts -= 1;
            let hi = self.raw_index(self.nelts);
            self.elts.get_mut(hi).take()
        } else {
            None
        }
    }
}

impl<T> Default for RingBuf<T> {
    #[inline]
    fn default() -> RingBuf<T> { RingBuf::new() }
}

impl<T> RingBuf<T> {
    /// Creates an empty `RingBuf`.
    pub fn new() -> RingBuf<T> {
        RingBuf::with_capacity(INITIAL_CAPACITY)
    }

    /// Creates an empty `RingBuf` with space for at least `n` elements.
    pub fn with_capacity(n: uint) -> RingBuf<T> {
        RingBuf{nelts: 0, lo: 0,
              elts: Vec::from_fn(cmp::max(MINIMUM_CAPACITY, n), |_| None)}
    }

    /// Retrieves an element in the `RingBuf` by index.
    ///
    /// Fails if there is no element with the given index.
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
    /// *buf.get_mut(1) = 7;
    /// assert_eq!(buf[1], 7);
    /// ```
    pub fn get_mut<'a>(&'a mut self, i: uint) -> &'a mut T {
        let idx = self.raw_index(i);
        match *self.elts.get_mut(idx) {
            None => fail!(),
            Some(ref mut v) => v
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
        let ri = self.raw_index(i);
        let rj = self.raw_index(j);
        self.elts.as_mut_slice().swap(ri, rj);
    }

    /// Returns the index in the underlying `Vec` for a given logical element
    /// index.
    fn raw_index(&self, idx: uint) -> uint {
        raw_index(self.lo, self.elts.len(), idx)
    }

    /// Reserves capacity for exactly `n` elements in the given `RingBuf`,
    /// doing nothing if `self`'s capacity is already equal to or greater
    /// than the requested capacity.
    pub fn reserve_exact(&mut self, n: uint) {
        self.elts.reserve_exact(n);
    }

    /// Reserves capacity for at least `n` elements in the given `RingBuf`,
    /// over-allocating in case the caller needs to reserve additional
    /// space.
    ///
    /// Do nothing if `self`'s capacity is already equal to or greater
    /// than the requested capacity.
    pub fn reserve(&mut self, n: uint) {
        self.elts.reserve(n);
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
    elts.reserve(nelts * 2);
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
            None => fail!(),
            Some(ref v) => v,
        }
    }
}

// FIXME(#12825) Indexing will always try IndexMut first and that causes issues.
/*impl<A> IndexMut<uint, A> for RingBuf<A> {
    #[inline]
    fn index_mut<'a>(&'a mut self, index: &uint) -> &'a mut A {
        self.get_mut(*index)
    }
}*/

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
    fn bench_push_back(b: &mut test::Bencher) {
        let mut deq = RingBuf::new();
        b.iter(|| {
            deq.push(0i);
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
    fn test_reserve_exact() {
        let mut d = RingBuf::new();
        d.push(0u64);
        d.reserve_exact(50);
        assert_eq!(d.elts.capacity(), 50);
        let mut d = RingBuf::new();
        d.push(0u32);
        d.reserve_exact(50);
        assert_eq!(d.elts.capacity(), 50);
    }

    #[test]
    fn test_reserve() {
        let mut d = RingBuf::new();
        d.push(0u64);
        d.reserve(50);
        assert_eq!(d.elts.capacity(), 64);
        let mut d = RingBuf::new();
        d.push(0u32);
        d.reserve(50);
        assert_eq!(d.elts.capacity(), 64);
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
}
