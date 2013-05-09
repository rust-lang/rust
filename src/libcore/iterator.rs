// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Composable external iterators

The `Iterator` trait defines an interface for objects which implement iteration as a state machine.

Algorithms like `zip` are provided as `Iterator` implementations which wrap other objects
implementing the `Iterator` trait.

*/

use prelude::*;
use util;

pub trait Iterator<A> {
    /// Advance the iterator and return the next value. Return `None` when the end is reached.
    fn next(&mut self) -> Option<A>;
}

/// Iterator adaptors provided for every `Iterator` implementation. The adaptor objects are also
/// implementations of the `Iterator` trait.
///
/// In the future these will be default methods instead of a utility trait.
pub trait IteratorUtil<A> {
    fn chain<U: Iterator<A>>(self, other: U) -> ChainIterator<Self, U>;
    fn zip<B, U: Iterator<B>>(self, other: U) -> ZipIterator<Self, U>;
    // FIXME: #5898: should be called map
    fn transform<'r, B>(self, f: &'r fn(A) -> B) -> MapIterator<'r, A, B, Self>;
    fn filter<'r>(self, predicate: &'r fn(&A) -> bool) -> FilterIterator<'r, A, Self>;
    fn enumerate(self) -> EnumerateIterator<Self>;
    fn skip_while<'r>(self, predicate: &'r fn(&A) -> bool) -> SkipWhileIterator<'r, A, Self>;
    fn take_while<'r>(self, predicate: &'r fn(&A) -> bool) -> TakeWhileIterator<'r, A, Self>;
    fn skip(self, n: uint) -> SkipIterator<Self>;
    fn take(self, n: uint) -> TakeIterator<Self>;

    fn scan<'r, St, B>(self, initial_state: St, f: &'r fn(&mut St, A) -> Option<B>)
        -> ScanIterator<'r, A, B, Self, St>;
    fn advance(&mut self, f: &fn(A) -> bool);

    // Ordered iterators
    // FIXME: #5898: shouldn't have the 'i' suffix
    fn intersecti<U: Iterator<A>>(self, other: U) -> Intersect<A, Self, U>;
    fn unioni<U: Iterator<A>>(self, other: U) -> Union<A, Self, U>;
    fn xori<U: Iterator<A>>(self, other: U) -> Xor<A, Self, U>;
    fn differencei<U: Iterator<A>>(self, other: U) -> Difference<A, Self, U>;
}

/// Iterator adaptors provided for every `Iterator` implementation. The adaptor objects are also
/// implementations of the `Iterator` trait.
///
/// In the future these will be default methods instead of a utility trait.
impl<A, T: Iterator<A>> IteratorUtil<A> for T {
    #[inline(always)]
    fn chain<U: Iterator<A>>(self, other: U) -> ChainIterator<T, U> {
        ChainIterator{a: self, b: other, flag: false}
    }

    #[inline(always)]
    fn zip<B, U: Iterator<B>>(self, other: U) -> ZipIterator<T, U> {
        ZipIterator{a: self, b: other}
    }

    // FIXME: #5898: should be called map
    #[inline(always)]
    fn transform<'r, B>(self, f: &'r fn(A) -> B) -> MapIterator<'r, A, B, T> {
        MapIterator{iter: self, f: f}
    }

    #[inline(always)]
    fn filter<'r>(self, predicate: &'r fn(&A) -> bool) -> FilterIterator<'r, A, T> {
        FilterIterator{iter: self, predicate: predicate}
    }

    #[inline(always)]
    fn enumerate(self) -> EnumerateIterator<T> {
        EnumerateIterator{iter: self, count: 0}
    }

    #[inline(always)]
    fn skip_while<'r>(self, predicate: &'r fn(&A) -> bool) -> SkipWhileIterator<'r, A, T> {
        SkipWhileIterator{iter: self, flag: false, predicate: predicate}
    }

    #[inline(always)]
    fn take_while<'r>(self, predicate: &'r fn(&A) -> bool) -> TakeWhileIterator<'r, A, T> {
        TakeWhileIterator{iter: self, flag: false, predicate: predicate}
    }

    #[inline(always)]
    fn skip(self, n: uint) -> SkipIterator<T> {
        SkipIterator{iter: self, n: n}
    }

    #[inline(always)]
    fn take(self, n: uint) -> TakeIterator<T> {
        TakeIterator{iter: self, n: n}
    }

    #[inline(always)]
    fn scan<'r, St, B>(self, initial_state: St, f: &'r fn(&mut St, A) -> Option<B>)
        -> ScanIterator<'r, A, B, T, St> {
        ScanIterator{iter: self, f: f, state: initial_state}
    }

    /// A shim implementing the `for` loop iteration protocol for iterator objects
    #[inline]
    fn advance(&mut self, f: &fn(A) -> bool) {
        loop {
            match self.next() {
                Some(x) => {
                    if !f(x) { return }
                }
                None => return
            }
        }
    }

    /// Returns an interator which will iterate over the intersection of `self`
    /// and `other`. This requires that the two iterators iterate over their
    /// values in ascending order.
    #[inline(always)]
    fn intersecti<U: Iterator<A>>(self, mut other: U) -> Intersect<A, T, U> {
        let mut self = self;
        let a = self.next();
        let b = other.next();
        Intersect{ a: a, b: b, i1: self, i2: other }
    }

    /// Returns an interator which will iterate over the union of `self` and
    /// `other`. This requires that the two iterators iterate over their values
    /// in ascending order.
    #[inline(always)]
    fn unioni<U: Iterator<A>>(self, mut other: U) -> Union<A, T, U> {
        let mut self = self;
        let a = self.next();
        let b = other.next();
        Union{ a: a, b: b, i1: self, i2: other }
    }

    /// Returns an interator which will iterate over the symmetric difference of
    /// `self` and `other`, or all values which are present in one, but not both
    /// iterators. This requires that the two iterators iterate over their
    /// values in ascending order.
    #[inline(always)]
    fn xori<U: Iterator<A>>(self, mut other: U) -> Xor<A, T, U> {
        let mut self = self;
        let a = self.next();
        let b = other.next();
        Xor{ a: a, b: b, i1: self, i2: other }
    }

    /// Returns an interator which will iterate over the difference of `self`
    /// and `other`, or all values which are present in `self`, but not in
    /// `other` iterators. This requires that the two iterators iterate over
    /// their value in ascending order.
    #[inline(always)]
    fn differencei<U: Iterator<A>>(self, mut other: U) -> Difference<A, T, U> {
        let mut self = self;
        let a = self.next();
        let b = other.next();
        Difference{ a: a, b: b, i1: self, i2: other }
    }
}

pub struct ChainIterator<T, U> {
    priv a: T,
    priv b: U,
    priv flag: bool
}

impl<A, T: Iterator<A>, U: Iterator<A>> Iterator<A> for ChainIterator<T, U> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        if self.flag {
            self.b.next()
        } else {
            match self.a.next() {
                Some(x) => return Some(x),
                _ => ()
            }
            self.flag = true;
            self.b.next()
        }
    }
}

pub struct ZipIterator<T, U> {
    priv a: T,
    priv b: U
}

impl<A, B, T: Iterator<A>, U: Iterator<B>> Iterator<(A, B)> for ZipIterator<T, U> {
    #[inline]
    fn next(&mut self) -> Option<(A, B)> {
        match (self.a.next(), self.b.next()) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None
        }
    }
}

pub struct MapIterator<'self, A, B, T> {
    priv iter: T,
    priv f: &'self fn(A) -> B
}

impl<'self, A, B, T: Iterator<A>> Iterator<B> for MapIterator<'self, A, B, T> {
    #[inline]
    fn next(&mut self) -> Option<B> {
        match self.iter.next() {
            Some(a) => Some((self.f)(a)),
            _ => None
        }
    }
}

pub struct FilterIterator<'self, A, T> {
    priv iter: T,
    priv predicate: &'self fn(&A) -> bool
}

impl<'self, A, T: Iterator<A>> Iterator<A> for FilterIterator<'self, A, T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        for self.iter.advance |x| {
            if (self.predicate)(&x) {
                return Some(x);
            } else {
                loop
            }
        }
        None
    }
}

pub struct EnumerateIterator<T> {
    priv iter: T,
    priv count: uint
}

impl<A, T: Iterator<A>> Iterator<(uint, A)> for EnumerateIterator<T> {
    #[inline]
    fn next(&mut self) -> Option<(uint, A)> {
        match self.iter.next() {
            Some(a) => {
                let ret = Some((self.count, a));
                self.count += 1;
                ret
            }
            _ => None
        }
    }
}

pub struct SkipWhileIterator<'self, A, T> {
    priv iter: T,
    priv flag: bool,
    priv predicate: &'self fn(&A) -> bool
}

impl<'self, A, T: Iterator<A>> Iterator<A> for SkipWhileIterator<'self, A, T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        let mut next = self.iter.next();
        if self.flag {
            next
        } else {
            loop {
                match next {
                    Some(x) => {
                        if (self.predicate)(&x) {
                            next = self.iter.next();
                            loop
                        } else {
                            self.flag = true;
                            return Some(x)
                        }
                    }
                    None => return None
                }
            }
        }
    }
}

pub struct TakeWhileIterator<'self, A, T> {
    priv iter: T,
    priv flag: bool,
    priv predicate: &'self fn(&A) -> bool
}

impl<'self, A, T: Iterator<A>> Iterator<A> for TakeWhileIterator<'self, A, T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        if self.flag {
            None
        } else {
            match self.iter.next() {
                Some(x) => {
                    if (self.predicate)(&x) {
                        Some(x)
                    } else {
                        self.flag = true;
                        None
                    }
                }
                None => None
            }
        }
    }
}

pub struct SkipIterator<T> {
    priv iter: T,
    priv n: uint
}

impl<A, T: Iterator<A>> Iterator<A> for SkipIterator<T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        let mut next = self.iter.next();
        if self.n == 0 {
            next
        } else {
            let n = self.n;
            for n.times {
                match next {
                    Some(_) => {
                        next = self.iter.next();
                        loop
                    }
                    None => {
                        self.n = 0;
                        return None
                    }
                }
            }
            self.n = 0;
            next
        }
    }
}

pub struct TakeIterator<T> {
    priv iter: T,
    priv n: uint
}

impl<A, T: Iterator<A>> Iterator<A> for TakeIterator<T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        let next = self.iter.next();
        if self.n != 0 {
            self.n -= 1;
            next
        } else {
            None
        }
    }
}

pub struct ScanIterator<'self, A, B, T, St> {
    priv iter: T,
    priv f: &'self fn(&mut St, A) -> Option<B>,
    state: St
}

impl<'self, A, B, T: Iterator<A>, St> Iterator<B> for ScanIterator<'self, A, B, T, St> {
    #[inline]
    fn next(&mut self) -> Option<B> {
        self.iter.next().chain(|a| (self.f)(&mut self.state, a))
    }
}

pub struct UnfoldrIterator<'self, A, St> {
    priv f: &'self fn(&mut St) -> Option<A>,
    state: St
}

pub impl<'self, A, St> UnfoldrIterator<'self, A, St> {
    #[inline]
    fn new(f: &'self fn(&mut St) -> Option<A>, initial_state: St)
        -> UnfoldrIterator<'self, A, St> {
        UnfoldrIterator {
            f: f,
            state: initial_state
        }
    }
}

impl<'self, A, St> Iterator<A> for UnfoldrIterator<'self, A, St> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        (self.f)(&mut self.state)
    }
}

pub struct Intersect<A, T, U> {
    priv i1: T,
    priv i2: U,
    priv a: Option<A>,
    priv b: Option<A>,
}

impl<A: TotalOrd, T: Iterator<A>, U: Iterator<A>> Iterator<A> for
    Intersect<A, T, U>
{
    #[inline]
    fn next(&mut self) -> Option<A> {
        loop {
            let cmp = match (&self.a, &self.b) {
                (&Some(ref a), &Some(ref b)) => a.cmp(b),
                _ => return None
            };

            match cmp {
                Less =>    { util::replace(&mut self.a, self.i1.next()); }
                Greater => { util::replace(&mut self.b, self.i2.next()); }
                Equal => {
                    util::replace(&mut self.a, self.i1.next());
                    return util::replace(&mut self.b, self.i2.next());
                }
            }
        }
    }
}

pub struct Union<A, T, U> {
    priv i1: T,
    priv i2: U,
    priv a: Option<A>,
    priv b: Option<A>,
}

impl<A: TotalOrd, T: Iterator<A>, U: Iterator<A>> Iterator<A> for Union<A, T, U>
{
    #[inline]
    fn next(&mut self) -> Option<A> {
        if self.a.is_none() && self.b.is_none() {
            return None;
        } else if self.a.is_none() {
            return util::replace(&mut self.b, self.i2.next());
        } else if self.b.is_none() {
            return util::replace(&mut self.a, self.i1.next());
        }
        let cmp = match (&self.a, &self.b) {
            (&Some(ref a), &Some(ref b)) => a.cmp(b),
            _ => fail!()
        };
        match cmp {
            Less =>    { return util::replace(&mut self.a, self.i1.next()); }
            Greater => { return util::replace(&mut self.b, self.i2.next()); }
            Equal => {
                util::replace(&mut self.b, self.i2.next());
                return util::replace(&mut self.a, self.i1.next());
            }
        }
    }
}

pub struct Xor<A, T, U> {
    priv i1: T,
    priv i2: U,
    priv a: Option<A>,
    priv b: Option<A>,
}

impl<A: TotalOrd, T: Iterator<A>, U: Iterator<A>> Iterator<A> for Xor<A, T, U> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        loop {
            if self.a.is_none() && self.b.is_none() {
                return None;
            } else if self.a.is_none() {
                return util::replace(&mut self.b, self.i2.next());
            } else if self.b.is_none() {
                return util::replace(&mut self.a, self.i1.next());
            }
            let cmp = match (&self.a, &self.b) {
                (&Some(ref a), &Some(ref b)) => a.cmp(b),
                _ => fail!()
            };
            match cmp {
                Less =>    { return util::replace(&mut self.a, self.i1.next()); }
                Greater => { return util::replace(&mut self.b, self.i2.next()); }
                Equal => {
                    util::replace(&mut self.a, self.i1.next());
                    util::replace(&mut self.b, self.i2.next());
                }
            }
        }
    }
}

pub struct Difference<A, T, U> {
    priv i1: T,
    priv i2: U,
    priv a: Option<A>,
    priv b: Option<A>,
}

impl<A: TotalOrd, T: Iterator<A>, U: Iterator<A>> Iterator<A> for
    Difference<A, T, U>
{
    #[inline]
    fn next(&mut self) -> Option<A> {
        loop {
            if self.a.is_none() {
                return None;
            } else if self.b.is_none() {
                return util::replace(&mut self.a, self.i1.next());
            }
            let cmp = match (&self.a, &self.b) {
                (&Some(ref a), &Some(ref b)) => a.cmp(b),
                _ => fail!()
            };
            match cmp {
                Less =>    { return util::replace(&mut self.a, self.i1.next()); }
                Greater => { util::replace(&mut self.b, self.i2.next()); }
                Equal => {
                    util::replace(&mut self.b, self.i2.next());
                    util::replace(&mut self.a, self.i1.next());
                }
            }
        }
    }
}

/// An infinite iterator starting at `start` and advancing by `step` with each iteration
pub struct Counter<A> {
    state: A,
    step: A
}

pub impl<A> Counter<A> {
    #[inline(always)]
    fn new(start: A, step: A) -> Counter<A> {
        Counter{state: start, step: step}
    }
}

impl<A: Add<A, A> + Clone> Iterator<A> for Counter<A> {
    #[inline(always)]
    fn next(&mut self) -> Option<A> {
        let result = self.state.clone();
        self.state = self.state.add(&self.step); // FIXME: #6050
        Some(result)
    }
}

/// An interator which will iterate over a binary search tree. This is
/// parameterized over functions which take a node and return the left and right
/// children. It is required that the nodes form a tree which is a binary search
/// tree
pub struct BSTIterator<'self, N, T> {
    priv f:     &'self fn(&'self N) -> T,
    priv left:  &'self fn(&'self N) -> &'self Option<~N>,
    priv right: &'self fn(&'self N) -> &'self Option<~N>,
    priv stack: ~[&'self ~N],
    priv node:  &'self Option<~N>,
}

pub impl<'self, N, T> BSTIterator<'self, N, T> {
    /// Creates a new iterator given the root node of a tree. The provided
    /// functions are used for yielding iterator values and acquiring the left
    /// and right children.
    #[inline(always)]
    fn new<'a>(root: &'a Option<~N>,
               f: &'a fn(&'a N) -> T,
               left: &'a fn(&'a N) -> &'a Option<~N>,
               right: &'a fn(&'a N) -> &'a Option<~N>) -> BSTIterator<'a, N, T>
    {
        BSTIterator{ f: f, stack: ~[], node: root, left: left, right: right }
    }
}

impl<'self, N, T> Iterator<T> for BSTIterator<'self, N, T> {
    /// Advance the iterator to the next node (in order) and return a
    /// tuple with a reference to the key and value. If there are no
    /// more nodes, return `None`.
    fn next(&mut self) -> Option<T> {
        while !self.stack.is_empty() || self.node.is_some() {
            match *self.node {
              Some(ref x) => {
                self.stack.push(x);
                self.node = (self.left)(*x);
              }
              None => {
                let res = self.stack.pop();
                self.node = (self.right)(*res);
                return Some((self.f)(*res));
              }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prelude::*;

    #[test]
    fn test_counter_to_vec() {
        let mut it = Counter::new(0, 5).take(10);
        let xs = iter::iter_to_vec(|f| it.advance(f));
        assert_eq!(xs, ~[0, 5, 10, 15, 20, 25, 30, 35, 40, 45]);
    }

    #[test]
    fn test_iterator_chain() {
        let xs = [0u, 1, 2, 3, 4, 5];
        let ys = [30u, 40, 50, 60];
        let expected = [0, 1, 2, 3, 4, 5, 30, 40, 50, 60];
        let mut it = xs.iter().chain(ys.iter());
        let mut i = 0;
        for it.advance |&x: &uint| {
            assert_eq!(x, expected[i]);
            i += 1;
        }
        assert_eq!(i, expected.len());

        let ys = Counter::new(30u, 10).take(4);
        let mut it = xs.iter().transform(|&x| x).chain(ys);
        let mut i = 0;
        for it.advance |x: uint| {
            assert_eq!(x, expected[i]);
            i += 1;
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_iterator_enumerate() {
        let xs = [0u, 1, 2, 3, 4, 5];
        let mut it = xs.iter().enumerate();
        for it.advance |(i, &x): (uint, &uint)| {
            assert_eq!(i, x);
        }
    }

    #[test]
    fn test_iterator_take_while() {
        let xs = [0u, 1, 2, 3, 5, 13, 15, 16, 17, 19];
        let ys = [0u, 1, 2, 3, 5, 13];
        let mut it = xs.iter().take_while(|&x| *x < 15u);
        let mut i = 0;
        for it.advance |&x: &uint| {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, ys.len());
    }

    #[test]
    fn test_iterator_skip_while() {
        let xs = [0u, 1, 2, 3, 5, 13, 15, 16, 17, 19];
        let ys = [15, 16, 17, 19];
        let mut it = xs.iter().skip_while(|&x| *x < 15u);
        let mut i = 0;
        for it.advance |&x: &uint| {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, ys.len());
    }

    #[test]
    fn test_iterator_skip() {
        let xs = [0u, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];
        let ys = [13, 15, 16, 17, 19, 20, 30];
        let mut it = xs.iter().skip(5);
        let mut i = 0;
        for it.advance |&x: &uint| {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, ys.len());
    }

    #[test]
    fn test_iterator_take() {
        let xs = [0u, 1, 2, 3, 5, 13, 15, 16, 17, 19];
        let ys = [0u, 1, 2, 3, 5];
        let mut it = xs.iter().take(5);
        let mut i = 0;
        for it.advance |&x: &uint| {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, ys.len());
    }

    #[test]
    fn test_iterator_scan() {
        // test the type inference
        fn add(old: &mut int, new: &uint) -> Option<float> {
            *old += *new as int;
            Some(*old as float)
        }
        let xs = [0u, 1, 2, 3, 4];
        let ys = [0f, 1f, 3f, 6f, 10f];

        let mut it = xs.iter().scan(0, add);
        let mut i = 0;
        for it.advance |x| {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, ys.len());
    }

    #[test]
    fn test_unfoldr() {
        fn count(st: &mut uint) -> Option<uint> {
            if *st < 10 {
                let ret = Some(*st);
                *st += 1;
                ret
            } else {
                None
            }
        }

        let mut it = UnfoldrIterator::new(count, 0);
        let mut i = 0;
        for it.advance |counted| {
            assert_eq!(counted, i);
            i += 1;
        }
        assert_eq!(i, 10);
    }
}
