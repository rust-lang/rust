// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A double-ended queue implemented as a circular buffer
//!
//! RingBuf implements the trait Deque. It should be imported with `use
//! collections::Deque`.

use core::prelude::*;

use core::cmp;
use core::default::Default;
use core::fmt;
use core::iter::RandomAccessIterator;

use {Deque, Collection, Mutable};
use vec::Vec;

static INITIAL_CAPACITY: uint = 8u; // 2^3
static MINIMUM_CAPACITY: uint = 2u;

/// RingBuf is a circular buffer that implements Deque.
#[deriving(Clone)]
pub struct RingBuf<T> {
    nelts: uint,
    lo: uint,
    elts: Vec<Option<T>>
}

impl<T> Collection for RingBuf<T> {
    /// Return the number of elements in the RingBuf
    fn len(&self) -> uint { self.nelts }
}

impl<T> Mutable for RingBuf<T> {
    /// Clear the RingBuf, removing all values.
    fn clear(&mut self) {
        for x in self.elts.mut_iter() { *x = None }
        self.nelts = 0;
        self.lo = 0;
    }
}

impl<T> Deque<T> for RingBuf<T> {
    /// Return a reference to the first element in the RingBuf
    fn front<'a>(&'a self) -> Option<&'a T> {
        if self.nelts > 0 { Some(self.get(0)) } else { None }
    }

    /// Return a mutable reference to the first element in the RingBuf
    fn front_mut<'a>(&'a mut self) -> Option<&'a mut T> {
        if self.nelts > 0 { Some(self.get_mut(0)) } else { None }
    }

    /// Return a reference to the last element in the RingBuf
    fn back<'a>(&'a self) -> Option<&'a T> {
        if self.nelts > 0 { Some(self.get(self.nelts - 1)) } else { None }
    }

    /// Return a mutable reference to the last element in the RingBuf
    fn back_mut<'a>(&'a mut self) -> Option<&'a mut T> {
        let nelts = self.nelts;
        if nelts > 0 { Some(self.get_mut(nelts - 1)) } else { None }
    }

    /// Remove and return the first element in the RingBuf, or None if it is empty
    fn pop_front(&mut self) -> Option<T> {
        let result = self.elts.get_mut(self.lo).take();
        if result.is_some() {
            self.lo = (self.lo + 1u) % self.elts.len();
            self.nelts -= 1u;
        }
        result
    }

    /// Remove and return the last element in the RingBuf, or None if it is empty
    fn pop_back(&mut self) -> Option<T> {
        if self.nelts > 0 {
            self.nelts -= 1;
            let hi = self.raw_index(self.nelts);
            self.elts.get_mut(hi).take()
        } else {
            None
        }
    }

    /// Prepend an element to the RingBuf
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

    /// Append an element to the RingBuf
    fn push_back(&mut self, t: T) {
        if self.nelts == self.elts.len() {
            grow(self.nelts, &mut self.lo, &mut self.elts);
        }
        let hi = self.raw_index(self.nelts);
        *self.elts.get_mut(hi) = Some(t);
        self.nelts += 1u;
    }
}

impl<T> Default for RingBuf<T> {
    #[inline]
    fn default() -> RingBuf<T> { RingBuf::new() }
}

impl<T> RingBuf<T> {
    /// Create an empty RingBuf
    pub fn new() -> RingBuf<T> {
        RingBuf::with_capacity(INITIAL_CAPACITY)
    }

    /// Create an empty RingBuf with space for at least `n` elements.
    pub fn with_capacity(n: uint) -> RingBuf<T> {
        RingBuf{nelts: 0, lo: 0,
              elts: Vec::from_fn(cmp::max(MINIMUM_CAPACITY, n), |_| None)}
    }

    /// Retrieve an element in the RingBuf by index
    ///
    /// Fails if there is no element with the given index
    pub fn get<'a>(&'a self, i: uint) -> &'a T {
        let idx = self.raw_index(i);
        match *self.elts.get(idx) {
            None => fail!(),
            Some(ref v) => v
        }
    }

    /// Retrieve an element in the RingBuf by index
    ///
    /// Fails if there is no element with the given index
    pub fn get_mut<'a>(&'a mut self, i: uint) -> &'a mut T {
        let idx = self.raw_index(i);
        match *self.elts.get_mut(idx) {
            None => fail!(),
            Some(ref mut v) => v
        }
    }

    /// Swap elements at indices `i` and `j`
    ///
    /// `i` and `j` may be equal.
    ///
    /// Fails if there is no element with the given index
    pub fn swap(&mut self, i: uint, j: uint) {
        assert!(i < self.len());
        assert!(j < self.len());
        let ri = self.raw_index(i);
        let rj = self.raw_index(j);
        self.elts.as_mut_slice().swap(ri, rj);
    }

    /// Return index in underlying vec for a given logical element index
    fn raw_index(&self, idx: uint) -> uint {
        raw_index(self.lo, self.elts.len(), idx)
    }

    /// Reserve capacity for exactly `n` elements in the given RingBuf,
    /// doing nothing if `self`'s capacity is already equal to or greater
    /// than the requested capacity
    ///
    /// # Arguments
    ///
    /// * n - The number of elements to reserve space for
    pub fn reserve_exact(&mut self, n: uint) {
        self.elts.reserve_exact(n);
    }

    /// Reserve capacity for at least `n` elements in the given RingBuf,
    /// over-allocating in case the caller needs to reserve additional
    /// space.
    ///
    /// Do nothing if `self`'s capacity is already equal to or greater
    /// than the requested capacity.
    ///
    /// # Arguments
    ///
    /// * n - The number of elements to reserve space for
    pub fn reserve(&mut self, n: uint) {
        self.elts.reserve(n);
    }

    /// Front-to-back iterator.
    pub fn iter<'a>(&'a self) -> Items<'a, T> {
        Items{index: 0, rindex: self.nelts, lo: self.lo, elts: self.elts.as_slice()}
    }

    /// Front-to-back iterator which returns mutable values.
    pub fn mut_iter<'a>(&'a mut self) -> MutItems<'a, T> {
        let start_index = raw_index(self.lo, self.elts.len(), 0);
        let end_index = raw_index(self.lo, self.elts.len(), self.nelts);

        // Divide up the array
        if end_index <= start_index {
            // Items to iterate goes from:
            //    start_index to self.elts.len()
            // and then
            //    0 to end_index
            let (temp, remaining1) = self.elts.mut_split_at(start_index);
            let (remaining2, _) = temp.mut_split_at(end_index);
            MutItems { remaining1: remaining1,
                                 remaining2: remaining2,
                                 nelts: self.nelts }
        } else {
            // Items to iterate goes from start_index to end_index:
            let (empty, elts) = self.elts.mut_split_at(0);
            let remaining1 = elts.mut_slice(start_index, end_index);
            MutItems { remaining1: remaining1,
                                 remaining2: empty,
                                 nelts: self.nelts }
        }
    }
}

/// RingBuf iterator
pub struct Items<'a, T> {
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
        Some(self.elts[raw_index].get_ref())
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
        Some(self.elts[raw_index].get_ref())
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
            Some(self.elts[raw_index].get_ref())
        }
    }
}

/// RingBuf mutable iterator
pub struct MutItems<'a, T> {
    remaining1: &'a mut [Option<T>],
    remaining2: &'a mut [Option<T>],
    nelts: uint,
}

impl<'a, T> Iterator<&'a mut T> for MutItems<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        if self.nelts == 0 {
            return None;
        }
        let r = if self.remaining1.len() > 0 {
            &mut self.remaining1
        } else {
            assert!(self.remaining2.len() > 0);
            &mut self.remaining2
        };
        self.nelts -= 1;
        Some(r.mut_shift_ref().unwrap().get_mut_ref())
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
        let r = if self.remaining2.len() > 0 {
            &mut self.remaining2
        } else {
            assert!(self.remaining1.len() > 0);
            &mut self.remaining1
        };
        self.nelts -= 1;
        Some(r.mut_pop_ref().unwrap().get_mut_ref())
    }
}

impl<'a, T> ExactSize<&'a mut T> for MutItems<'a, T> {}

/// Grow is only called on full elts, so nelts is also len(elts), unlike
/// elsewhere.
fn grow<T>(nelts: uint, loptr: &mut uint, elts: &mut Vec<Option<T>>) {
    assert_eq!(nelts, elts.len());
    let lo = *loptr;
    let newlen = nelts * 2;
    elts.reserve(newlen);

    /* fill with None */
    for _ in range(elts.len(), elts.capacity()) {
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

/// Return index in underlying vec for a given logical element index
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
    use test::Bencher;
    use test;

    use {Deque, Mutable};
    use super::RingBuf;
    use vec::Vec;

    #[test]
    fn test_simple() {
        let mut d = RingBuf::new();
        assert_eq!(d.len(), 0u);
        d.push_front(17);
        d.push_front(42);
        d.push_back(137);
        assert_eq!(d.len(), 3u);
        d.push_back(137);
        assert_eq!(d.len(), 4u);
        debug!("{:?}", d.front());
        assert_eq!(*d.front().unwrap(), 42);
        debug!("{:?}", d.back());
        assert_eq!(*d.back().unwrap(), 137);
        let mut i = d.pop_front();
        debug!("{:?}", i);
        assert_eq!(i, Some(42));
        i = d.pop_back();
        debug!("{:?}", i);
        assert_eq!(i, Some(137));
        i = d.pop_back();
        debug!("{:?}", i);
        assert_eq!(i, Some(137));
        i = d.pop_back();
        debug!("{:?}", i);
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
        debug!("{:?}", d.get(0));
        debug!("{:?}", d.get(1));
        debug!("{:?}", d.get(2));
        debug!("{:?}", d.get(3));
        assert_eq!(*d.get(0), 1);
        assert_eq!(*d.get(1), 2);
        assert_eq!(*d.get(2), 3);
        assert_eq!(*d.get(3), 4);
    }

    #[test]
    fn test_boxes() {
        let a: @int = @5;
        let b: @int = @72;
        let c: @int = @64;
        let d: @int = @175;

        let mut deq = RingBuf::new();
        assert_eq!(deq.len(), 0);
        deq.push_front(a);
        deq.push_front(b);
        deq.push_back(c);
        assert_eq!(deq.len(), 3);
        deq.push_back(d);
        assert_eq!(deq.len(), 4);
        assert_eq!(deq.front(), Some(&b));
        assert_eq!(deq.back(), Some(&d));
        assert_eq!(deq.pop_front(), Some(b));
        assert_eq!(deq.pop_back(), Some(d));
        assert_eq!(deq.pop_back(), Some(c));
        assert_eq!(deq.pop_back(), Some(a));
        assert_eq!(deq.len(), 0);
        deq.push_back(c);
        assert_eq!(deq.len(), 1);
        deq.push_front(b);
        assert_eq!(deq.len(), 2);
        deq.push_back(d);
        assert_eq!(deq.len(), 3);
        deq.push_front(a);
        assert_eq!(deq.len(), 4);
        assert_eq!(*deq.get(0), a);
        assert_eq!(*deq.get(1), b);
        assert_eq!(*deq.get(2), c);
        assert_eq!(*deq.get(3), d);
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
        assert_eq!((*deq.get(0)).clone(), a.clone());
        assert_eq!((*deq.get(1)).clone(), b.clone());
        assert_eq!((*deq.get(2)).clone(), c.clone());
        assert_eq!((*deq.get(3)).clone(), d.clone());
    }

    #[test]
    fn test_push_front_grow() {
        let mut deq = RingBuf::new();
        for i in range(0u, 66) {
            deq.push_front(i);
        }
        assert_eq!(deq.len(), 66);

        for i in range(0u, 66) {
            assert_eq!(*deq.get(i), 65 - i);
        }

        let mut deq = RingBuf::new();
        for i in range(0u, 66) {
            deq.push_back(i);
        }

        for i in range(0u, 66) {
            assert_eq!(*deq.get(i), i);
        }
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
            deq.push_back(0);
        })
    }

    #[bench]
    fn bench_push_front(b: &mut test::Bencher) {
        let mut deq = RingBuf::new();
        b.iter(|| {
            deq.push_front(0);
        })
    }

    #[bench]
    fn bench_grow(b: &mut test::Bencher) {
        let mut deq = RingBuf::new();
        b.iter(|| {
            for _ in range(0, 65) {
                deq.push_front(1);
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
    fn test_param_at_int() {
        test_parameterized::<@int>(@5, @72, @64, @175);
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
        d.push_back(1);
        assert_eq!(d.len(), 1);
        let mut d = RingBuf::with_capacity(50);
        d.push_back(1);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_reserve_exact() {
        let mut d = RingBuf::new();
        d.push_back(0u64);
        d.reserve_exact(50);
        assert_eq!(d.elts.capacity(), 50);
        let mut d = RingBuf::new();
        d.push_back(0u32);
        d.reserve_exact(50);
        assert_eq!(d.elts.capacity(), 50);
    }

    #[test]
    fn test_reserve() {
        let mut d = RingBuf::new();
        d.push_back(0u64);
        d.reserve(50);
        assert_eq!(d.elts.capacity(), 64);
        let mut d = RingBuf::new();
        d.push_back(0u32);
        d.reserve(50);
        assert_eq!(d.elts.capacity(), 64);
    }

    #[test]
    fn test_swap() {
        let mut d: RingBuf<int> = range(0, 5).collect();
        d.pop_front();
        d.swap(0, 3);
        assert_eq!(d.iter().map(|&x|x).collect::<Vec<int>>(), vec!(4, 2, 3, 1));
    }

    #[test]
    fn test_iter() {
        let mut d = RingBuf::new();
        assert_eq!(d.iter().next(), None);
        assert_eq!(d.iter().size_hint(), (0, Some(0)));

        for i in range(0, 5) {
            d.push_back(i);
        }
        assert_eq!(d.iter().collect::<Vec<&int>>().as_slice(), &[&0,&1,&2,&3,&4]);

        for i in range(6, 9) {
            d.push_front(i);
        }
        assert_eq!(d.iter().collect::<Vec<&int>>().as_slice(), &[&8,&7,&6,&0,&1,&2,&3,&4]);

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

        for i in range(0, 5) {
            d.push_back(i);
        }
        assert_eq!(d.iter().rev().collect::<Vec<&int>>().as_slice(), &[&4,&3,&2,&1,&0]);

        for i in range(6, 9) {
            d.push_front(i);
        }
        assert_eq!(d.iter().rev().collect::<Vec<&int>>().as_slice(), &[&4,&3,&2,&1,&0,&6,&7,&8]);
    }

    #[test]
    fn test_mut_rev_iter_wrap() {
        let mut d = RingBuf::with_capacity(3);
        assert!(d.mut_iter().rev().next().is_none());

        d.push_back(1);
        d.push_back(2);
        d.push_back(3);
        assert_eq!(d.pop_front(), Some(1));
        d.push_back(4);

        assert_eq!(d.mut_iter().rev().map(|x| *x).collect::<Vec<int>>(),
                   vec!(4, 3, 2));
    }

    #[test]
    fn test_mut_iter() {
        let mut d = RingBuf::new();
        assert!(d.mut_iter().next().is_none());

        for i in range(0u, 3) {
            d.push_front(i);
        }

        for (i, elt) in d.mut_iter().enumerate() {
            assert_eq!(*elt, 2 - i);
            *elt = i;
        }

        {
            let mut it = d.mut_iter();
            assert_eq!(*it.next().unwrap(), 0);
            assert_eq!(*it.next().unwrap(), 1);
            assert_eq!(*it.next().unwrap(), 2);
            assert!(it.next().is_none());
        }
    }

    #[test]
    fn test_mut_rev_iter() {
        let mut d = RingBuf::new();
        assert!(d.mut_iter().rev().next().is_none());

        for i in range(0u, 3) {
            d.push_front(i);
        }

        for (i, elt) in d.mut_iter().rev().enumerate() {
            assert_eq!(*elt, i);
            *elt = i;
        }

        {
            let mut it = d.mut_iter().rev();
            assert_eq!(*it.next().unwrap(), 0);
            assert_eq!(*it.next().unwrap(), 1);
            assert_eq!(*it.next().unwrap(), 2);
            assert!(it.next().is_none());
        }
    }

    #[test]
    fn test_from_iter() {
        use std::iter;
        let v = vec!(1,2,3,4,5,6,7);
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
        d.push_front(17);
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
        d.push_front(137);
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
    fn test_show() {
        let ringbuf: RingBuf<int> = range(0, 10).collect();
        assert!(format!("{}", ringbuf).as_slice() == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");

        let ringbuf: RingBuf<&str> = vec!["just", "one", "test", "more"].iter()
                                                                        .map(|&s| s)
                                                                        .collect();
        assert!(format!("{}", ringbuf).as_slice() == "[just, one, test, more]");
    }
}
