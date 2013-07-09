// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A double-ended queue implemented as a circular buffer

use std::num;
use std::uint;
use std::vec;
use std::iterator::FromIterator;

static INITIAL_CAPACITY: uint = 8u; // 2^3
static MINIMUM_CAPACITY: uint = 2u;

#[allow(missing_doc)]
#[deriving(Clone)]
pub struct Deque<T> {
    priv nelts: uint,
    priv lo: uint,
    priv elts: ~[Option<T>]
}

impl<T> Container for Deque<T> {
    /// Return the number of elements in the deque
    fn len(&self) -> uint { self.nelts }

    /// Return true if the deque contains no elements
    fn is_empty(&self) -> bool { self.len() == 0 }
}

impl<T> Mutable for Deque<T> {
    /// Clear the deque, removing all values.
    fn clear(&mut self) {
        for self.elts.mut_iter().advance |x| { *x = None }
        self.nelts = 0;
        self.lo = 0;
    }
}

impl<T> Deque<T> {
    /// Create an empty Deque
    pub fn new() -> Deque<T> {
        Deque::with_capacity(INITIAL_CAPACITY)
    }

    /// Create an empty Deque with space for at least `n` elements.
    pub fn with_capacity(n: uint) -> Deque<T> {
        Deque{nelts: 0, lo: 0,
              elts: vec::from_fn(num::max(MINIMUM_CAPACITY, n), |_| None)}
    }

    /// Return a reference to the first element in the deque
    ///
    /// Fails if the deque is empty
    pub fn peek_front<'a>(&'a self) -> &'a T { get(self.elts, self.raw_index(0)) }

    /// Return a reference to the last element in the deque
    ///
    /// Fails if the deque is empty
    pub fn peek_back<'a>(&'a self) -> &'a T {
        if self.nelts > 0 {
            get(self.elts, self.raw_index(self.nelts - 1))
        } else {
            fail!("peek_back: empty deque");
        }
    }

    /// Retrieve an element in the deque by index
    ///
    /// Fails if there is no element with the given index
    pub fn get<'a>(&'a self, i: int) -> &'a T {
        let idx = (self.lo + (i as uint)) % self.elts.len();
        get(self.elts, idx)
    }

    /// Remove and return the first element in the deque
    ///
    /// Fails if the deque is empty
    pub fn pop_front(&mut self) -> T {
        let result = self.elts[self.lo].swap_unwrap();
        self.lo = (self.lo + 1u) % self.elts.len();
        self.nelts -= 1u;
        result
    }

    /// Return index in underlying vec for a given logical element index
    fn raw_index(&self, idx: uint) -> uint {
        raw_index(self.lo, self.elts.len(), idx)
    }

    /// Remove and return the last element in the deque
    ///
    /// Fails if the deque is empty
    pub fn pop_back(&mut self) -> T {
        self.nelts -= 1;
        let hi = self.raw_index(self.nelts);
        self.elts[hi].swap_unwrap()
    }

    /// Prepend an element to the deque
    pub fn add_front(&mut self, t: T) {
        if self.nelts == self.elts.len() {
            grow(self.nelts, &mut self.lo, &mut self.elts);
        }
        if self.lo == 0u {
            self.lo = self.elts.len() - 1u;
        } else { self.lo -= 1u; }
        self.elts[self.lo] = Some(t);
        self.nelts += 1u;
    }

    /// Append an element to the deque
    pub fn add_back(&mut self, t: T) {
        if self.nelts == self.elts.len() {
            grow(self.nelts, &mut self.lo, &mut self.elts);
        }
        let hi = self.raw_index(self.nelts);
        self.elts[hi] = Some(t);
        self.nelts += 1u;
    }

    /// Reserve capacity for exactly `n` elements in the given deque,
    /// doing nothing if `self`'s capacity is already equal to or greater
    /// than the requested capacity
    ///
    /// # Arguments
    ///
    /// * n - The number of elements to reserve space for
    pub fn reserve(&mut self, n: uint) {
        self.elts.reserve(n);
    }

    /// Reserve capacity for at least `n` elements in the given deque,
    /// over-allocating in case the caller needs to reserve additional
    /// space.
    ///
    /// Do nothing if `self`'s capacity is already equal to or greater
    /// than the requested capacity.
    ///
    /// # Arguments
    ///
    /// * n - The number of elements to reserve space for
    pub fn reserve_at_least(&mut self, n: uint) {
        self.elts.reserve_at_least(n);
    }

    /// Front-to-back iterator.
    pub fn iter<'a>(&'a self) -> DequeIterator<'a, T> {
        DequeIterator{index: 0, nelts: self.nelts, elts: self.elts, lo: self.lo}
    }

    /// Front-to-back iterator which returns mutable values.
    pub fn mut_iter<'a>(&'a mut self) -> DequeMutIterator<'a, T> {
        DequeMutIterator{index: 0, nelts: self.nelts, elts: self.elts, lo: self.lo}
    }

    /// Back-to-front iterator.
    pub fn rev_iter<'a>(&'a self) -> DequeRevIterator<'a, T> {
        DequeRevIterator{index: self.nelts-1, nelts: self.nelts, elts: self.elts,
                         lo: self.lo}
    }

    /// Back-to-front iterator which returns mutable values.
    pub fn mut_rev_iter<'a>(&'a mut self) -> DequeMutRevIterator<'a, T> {
        DequeMutRevIterator{index: self.nelts-1, nelts: self.nelts, elts: self.elts,
                            lo: self.lo}
    }
}

macro_rules! iterator {
    (impl $name:ident -> $elem:ty, $getter:ident, $step:expr) => {
        impl<'self, T> Iterator<$elem> for $name<'self, T> {
            #[inline]
            fn next(&mut self) -> Option<$elem> {
                if self.nelts == 0 {
                    return None;
                }
                let raw_index = raw_index(self.lo, self.elts.len(), self.index);
                self.index += $step;
                self.nelts -= 1;
                Some(self.elts[raw_index]. $getter ())
            }
        }
    }
}

/// Deque iterator
pub struct DequeIterator<'self, T> {
    priv lo: uint,
    priv nelts: uint,
    priv index: uint,
    priv elts: &'self [Option<T>],
}
iterator!{impl DequeIterator -> &'self T, get_ref, 1}

/// Deque reverse iterator
pub struct DequeRevIterator<'self, T> {
    priv lo: uint,
    priv nelts: uint,
    priv index: uint,
    priv elts: &'self [Option<T>],
}
iterator!{impl DequeRevIterator -> &'self T, get_ref, -1}

/// Deque mutable iterator
pub struct DequeMutIterator<'self, T> {
    priv lo: uint,
    priv nelts: uint,
    priv index: uint,
    priv elts: &'self mut [Option<T>],
}
iterator!{impl DequeMutIterator -> &'self mut T, get_mut_ref, 1}

/// Deque mutable reverse iterator
pub struct DequeMutRevIterator<'self, T> {
    priv lo: uint,
    priv nelts: uint,
    priv index: uint,
    priv elts: &'self mut [Option<T>],
}
iterator!{impl DequeMutRevIterator -> &'self mut T, get_mut_ref, -1}

/// Grow is only called on full elts, so nelts is also len(elts), unlike
/// elsewhere.
fn grow<T>(nelts: uint, loptr: &mut uint, elts: &mut ~[Option<T>]) {
    assert_eq!(nelts, elts.len());
    let lo = *loptr;
    let newlen = nelts * 2;
    elts.reserve(newlen);

    /* fill with None */
    for uint::range(elts.len(), elts.capacity()) |_| {
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
        for uint::range(0, lo) |i| {
            elts.swap(i, nelts + i);
        }
    } else {                // B
        for uint::range(lo, nelts) |i| {
            elts.swap(i, newlen - nelts + i);
        }
        *loptr += newlen - nelts;
    }
}

fn get<'r, T>(elts: &'r [Option<T>], i: uint) -> &'r T {
    match elts[i] { Some(ref t) => t, _ => fail!() }
}

/// Return index in underlying vec for a given logical element index
fn raw_index(lo: uint, len: uint, index: uint) -> uint {
    if lo >= len - index {
        lo + index - len
    } else {
        lo + index
    }
}

impl<A: Eq> Eq for Deque<A> {
    fn eq(&self, other: &Deque<A>) -> bool {
        self.nelts == other.nelts &&
            self.iter().zip(other.iter()).all(|(a, b)| a.eq(b))
    }
    fn ne(&self, other: &Deque<A>) -> bool {
        !self.eq(other)
    }
}

impl<A, T: Iterator<A>> FromIterator<A, T> for Deque<A> {
    fn from_iterator(iterator: &mut T) -> Deque<A> {
        let mut deq = Deque::new();
        for iterator.advance |elt| {
            deq.add_back(elt);
        }
        deq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Eq;
    use std::kinds::Copy;
    use std::{int, uint};
    use extra::test;

    #[test]
    fn test_simple() {
        let mut d = Deque::new();
        assert_eq!(d.len(), 0u);
        d.add_front(17);
        d.add_front(42);
        d.add_back(137);
        assert_eq!(d.len(), 3u);
        d.add_back(137);
        assert_eq!(d.len(), 4u);
        debug!(d.peek_front());
        assert_eq!(*d.peek_front(), 42);
        debug!(d.peek_back());
        assert_eq!(*d.peek_back(), 137);
        let mut i: int = d.pop_front();
        debug!(i);
        assert_eq!(i, 42);
        i = d.pop_back();
        debug!(i);
        assert_eq!(i, 137);
        i = d.pop_back();
        debug!(i);
        assert_eq!(i, 137);
        i = d.pop_back();
        debug!(i);
        assert_eq!(i, 17);
        assert_eq!(d.len(), 0u);
        d.add_back(3);
        assert_eq!(d.len(), 1u);
        d.add_front(2);
        assert_eq!(d.len(), 2u);
        d.add_back(4);
        assert_eq!(d.len(), 3u);
        d.add_front(1);
        assert_eq!(d.len(), 4u);
        debug!(d.get(0));
        debug!(d.get(1));
        debug!(d.get(2));
        debug!(d.get(3));
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

        let mut deq = Deque::new();
        assert_eq!(deq.len(), 0);
        deq.add_front(a);
        deq.add_front(b);
        deq.add_back(c);
        assert_eq!(deq.len(), 3);
        deq.add_back(d);
        assert_eq!(deq.len(), 4);
        assert_eq!(*deq.peek_front(), b);
        assert_eq!(*deq.peek_back(), d);
        assert_eq!(deq.pop_front(), b);
        assert_eq!(deq.pop_back(), d);
        assert_eq!(deq.pop_back(), c);
        assert_eq!(deq.pop_back(), a);
        assert_eq!(deq.len(), 0);
        deq.add_back(c);
        assert_eq!(deq.len(), 1);
        deq.add_front(b);
        assert_eq!(deq.len(), 2);
        deq.add_back(d);
        assert_eq!(deq.len(), 3);
        deq.add_front(a);
        assert_eq!(deq.len(), 4);
        assert_eq!(*deq.get(0), a);
        assert_eq!(*deq.get(1), b);
        assert_eq!(*deq.get(2), c);
        assert_eq!(*deq.get(3), d);
    }

    #[cfg(test)]
    fn test_parameterized<T:Copy + Eq>(a: T, b: T, c: T, d: T) {
        let mut deq = Deque::new();
        assert_eq!(deq.len(), 0);
        deq.add_front(copy a);
        deq.add_front(copy b);
        deq.add_back(copy c);
        assert_eq!(deq.len(), 3);
        deq.add_back(copy d);
        assert_eq!(deq.len(), 4);
        assert_eq!(copy *deq.peek_front(), copy b);
        assert_eq!(copy *deq.peek_back(), copy d);
        assert_eq!(deq.pop_front(), copy b);
        assert_eq!(deq.pop_back(), copy d);
        assert_eq!(deq.pop_back(), copy c);
        assert_eq!(deq.pop_back(), copy a);
        assert_eq!(deq.len(), 0);
        deq.add_back(copy c);
        assert_eq!(deq.len(), 1);
        deq.add_front(copy b);
        assert_eq!(deq.len(), 2);
        deq.add_back(copy d);
        assert_eq!(deq.len(), 3);
        deq.add_front(copy a);
        assert_eq!(deq.len(), 4);
        assert_eq!(copy *deq.get(0), copy a);
        assert_eq!(copy *deq.get(1), copy b);
        assert_eq!(copy *deq.get(2), copy c);
        assert_eq!(copy *deq.get(3), copy d);
    }

    #[test]
    fn test_add_front_grow() {
        let mut deq = Deque::new();
        for int::range(0, 66) |i| {
            deq.add_front(i);
        }
        assert_eq!(deq.len(), 66);

        for int::range(0, 66) |i| {
            assert_eq!(*deq.get(i), 65 - i);
        }

        let mut deq = Deque::new();
        for int::range(0, 66) |i| {
            deq.add_back(i);
        }

        for int::range(0, 66) |i| {
            assert_eq!(*deq.get(i), i);
        }
    }

    #[bench]
    fn bench_new(b: &mut test::BenchHarness) {
        do b.iter {
            let _ = Deque::new::<u64>();
        }
    }

    #[bench]
    fn bench_add_back(b: &mut test::BenchHarness) {
        let mut deq = Deque::new();
        do b.iter {
            deq.add_back(0);
        }
    }

    #[bench]
    fn bench_add_front(b: &mut test::BenchHarness) {
        let mut deq = Deque::new();
        do b.iter {
            deq.add_front(0);
        }
    }

    #[bench]
    fn bench_grow(b: &mut test::BenchHarness) {
        let mut deq = Deque::new();
        do b.iter {
            for 65.times {
                deq.add_front(1);
            }
        }
    }

    #[deriving(Eq)]
    enum Taggy { One(int), Two(int, int), Three(int, int, int), }

    #[deriving(Eq)]
    enum Taggypar<T> {
        Onepar(int), Twopar(int, int), Threepar(int, int, int),
    }

    #[deriving(Eq)]
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
        let mut d = Deque::with_capacity(0);
        d.add_back(1);
        assert_eq!(d.len(), 1);
        let mut d = Deque::with_capacity(50);
        d.add_back(1);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_reserve() {
        let mut d = Deque::new();
        d.add_back(0u64);
        d.reserve(50);
        assert_eq!(d.elts.capacity(), 50);
        let mut d = Deque::new();
        d.add_back(0u32);
        d.reserve(50);
        assert_eq!(d.elts.capacity(), 50);
    }

    #[test]
    fn test_reserve_at_least() {
        let mut d = Deque::new();
        d.add_back(0u64);
        d.reserve_at_least(50);
        assert_eq!(d.elts.capacity(), 64);
        let mut d = Deque::new();
        d.add_back(0u32);
        d.reserve_at_least(50);
        assert_eq!(d.elts.capacity(), 64);
    }

    #[test]
    fn test_iter() {
        let mut d = Deque::new();
        assert_eq!(d.iter().next(), None);

        for int::range(0,5) |i| {
            d.add_back(i);
        }
        assert_eq!(d.iter().collect::<~[&int]>(), ~[&0,&1,&2,&3,&4]);

        for int::range(6,9) |i| {
            d.add_front(i);
        }
        assert_eq!(d.iter().collect::<~[&int]>(), ~[&8,&7,&6,&0,&1,&2,&3,&4]);
    }

    #[test]
    fn test_rev_iter() {
        let mut d = Deque::new();
        assert_eq!(d.rev_iter().next(), None);

        for int::range(0,5) |i| {
            d.add_back(i);
        }
        assert_eq!(d.rev_iter().collect::<~[&int]>(), ~[&4,&3,&2,&1,&0]);

        for int::range(6,9) |i| {
            d.add_front(i);
        }
        assert_eq!(d.rev_iter().collect::<~[&int]>(), ~[&4,&3,&2,&1,&0,&6,&7,&8]);
    }

    #[test]
    fn test_mut_iter() {
        let mut d = Deque::new();
        assert!(d.mut_iter().next().is_none());

        for uint::range(0,3) |i| {
            d.add_front(i);
        }

        for d.mut_iter().enumerate().advance |(i, elt)| {
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
        let mut d = Deque::new();
        assert!(d.mut_rev_iter().next().is_none());

        for uint::range(0,3) |i| {
            d.add_front(i);
        }

        for d.mut_rev_iter().enumerate().advance |(i, elt)| {
            assert_eq!(*elt, i);
            *elt = i;
        }

        {
            let mut it = d.mut_rev_iter();
            assert_eq!(*it.next().unwrap(), 0);
            assert_eq!(*it.next().unwrap(), 1);
            assert_eq!(*it.next().unwrap(), 2);
            assert!(it.next().is_none());
        }
    }

    #[test]
    fn test_from_iterator() {
        use std::iterator;
        let v = ~[1,2,3,4,5,6,7];
        let deq: Deque<int> = v.iter().transform(|&x| x).collect();
        let u: ~[int] = deq.iter().transform(|&x| x).collect();
        assert_eq!(u, v);

        let mut seq = iterator::Counter::new(0u, 2).take_(256);
        let deq: Deque<uint> = seq.collect();
        for deq.iter().enumerate().advance |(i, &x)| {
            assert_eq!(2*i, x);
        }
        assert_eq!(deq.len(), 256);
    }

    #[test]
    fn test_clone() {
        let mut d = Deque::new();
        d.add_front(17);
        d.add_front(42);
        d.add_back(137);
        d.add_back(137);
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
        let mut d = Deque::new();
        assert_eq!(&d, &Deque::with_capacity(0));
        d.add_front(137);
        d.add_front(17);
        d.add_front(42);
        d.add_back(137);
        let mut e = Deque::with_capacity(0);
        e.add_back(42);
        e.add_back(17);
        e.add_back(137);
        e.add_back(137);
        assert_eq!(&e, &d);
        e.pop_back();
        e.add_back(0);
        assert!(e != d);
        e.clear();
        assert_eq!(e, Deque::new());
    }
}
