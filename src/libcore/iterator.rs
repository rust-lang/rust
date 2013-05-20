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
use num::{Zero, One};

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
    fn filter_map<'r,  B>(self, f: &'r fn(A) -> Option<B>) -> FilterMapIterator<'r, A, B, Self>;
    fn enumerate(self) -> EnumerateIterator<Self>;
    fn skip_while<'r>(self, predicate: &'r fn(&A) -> bool) -> SkipWhileIterator<'r, A, Self>;
    fn take_while<'r>(self, predicate: &'r fn(&A) -> bool) -> TakeWhileIterator<'r, A, Self>;
    fn skip(self, n: uint) -> SkipIterator<Self>;
    fn take(self, n: uint) -> TakeIterator<Self>;
    fn scan<'r, St, B>(self, initial_state: St, f: &'r fn(&mut St, A) -> Option<B>)
        -> ScanIterator<'r, A, B, Self, St>;
    fn advance(&mut self, f: &fn(A) -> bool) -> bool;
    fn to_vec(&mut self) -> ~[A];
    fn nth(&mut self, n: uint) -> Option<A>;
    fn last(&mut self) -> Option<A>;
    fn fold<B>(&mut self, start: B, f: &fn(B, A) -> B) -> B;
    fn count(&mut self) -> uint;
    fn all(&mut self, f: &fn(&A) -> bool) -> bool;
    fn any(&mut self, f: &fn(&A) -> bool) -> bool;
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
    fn filter_map<'r, B>(self, f: &'r fn(A) -> Option<B>) -> FilterMapIterator<'r, A, B, T> {
        FilterMapIterator { iter: self, f: f }
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
    fn advance(&mut self, f: &fn(A) -> bool) -> bool {
        loop {
            match self.next() {
                Some(x) => {
                    if !f(x) { return false; }
                }
                None => { return true; }
            }
        }
    }

    #[inline(always)]
    fn to_vec(&mut self) -> ~[A] {
        iter::to_vec::<A>(|f| self.advance(f))
    }

    /// Return the `n`th item yielded by an iterator.
    #[inline(always)]
    fn nth(&mut self, mut n: uint) -> Option<A> {
        loop {
            match self.next() {
                Some(x) => if n == 0 { return Some(x) },
                None => return None
            }
            n -= 1;
        }
    }

    /// Return the last item yielded by an iterator.
    #[inline(always)]
    fn last(&mut self) -> Option<A> {
        let mut last = None;
        for self.advance |x| { last = Some(x); }
        last
    }

    /// Reduce an iterator to an accumulated value
    #[inline]
    fn fold<B>(&mut self, init: B, f: &fn(B, A) -> B) -> B {
        let mut accum = init;
        loop {
            match self.next() {
                Some(x) => { accum = f(accum, x); }
                None    => { break; }
            }
        }
        return accum;
    }

    /// Count the number of items yielded by an iterator
    #[inline(always)]
    fn count(&mut self) -> uint { self.fold(0, |cnt, _x| cnt + 1) }

    #[inline(always)]
    fn all(&mut self, f: &fn(&A) -> bool) -> bool {
        for self.advance |x| { if !f(&x) { return false; } }
        return true;
    }

    #[inline(always)]
    fn any(&mut self, f: &fn(&A) -> bool) -> bool {
        for self.advance |x| { if f(&x) { return true; } }
        return false;
    }
}

pub trait AdditiveIterator<A> {
    fn sum(&mut self) -> A;
}

impl<A: Add<A, A> + Zero, T: Iterator<A>> AdditiveIterator<A> for T {
    #[inline(always)]
    fn sum(&mut self) -> A { self.fold(Zero::zero::<A>(), |s, x| s + x) }
}

pub trait MultiplicativeIterator<A> {
    fn product(&mut self) -> A;
}

impl<A: Mul<A, A> + One, T: Iterator<A>> MultiplicativeIterator<A> for T {
    #[inline(always)]
    fn product(&mut self) -> A { self.fold(One::one::<A>(), |p, x| p * x) }
}

pub trait OrdIterator<A> {
    fn max(&mut self) -> Option<A>;
    fn min(&mut self) -> Option<A>;
}

impl<A: Ord, T: Iterator<A>> OrdIterator<A> for T {
    #[inline(always)]
    fn max(&mut self) -> Option<A> {
        self.fold(None, |max, x| {
            match max {
                None    => Some(x),
                Some(y) => Some(cmp::max(x, y))
            }
        })
    }

    #[inline(always)]
    fn min(&mut self) -> Option<A> {
        self.fold(None, |min, x| {
            match min {
                None    => Some(x),
                Some(y) => Some(cmp::min(x, y))
            }
        })
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

pub struct FilterMapIterator<'self, A, B, T> {
    priv iter: T,
    priv f: &'self fn(A) -> Option<B>
}

impl<'self, A, B, T: Iterator<A>> Iterator<B> for FilterMapIterator<'self, A, B, T> {
    #[inline]
    fn next(&mut self) -> Option<B> {
        for self.iter.advance |x| {
            match (self.f)(x) {
                Some(y) => return Some(y),
                None => ()
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

#[cfg(test)]
mod tests {
    use super::*;
    use prelude::*;

    #[test]
    fn test_counter_to_vec() {
        let mut it = Counter::new(0, 5).take(10);
        let xs = iter::to_vec(|f| it.advance(f));
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
    fn test_filter_map() {
        let mut it = Counter::new(0u, 1u).take(10)
            .filter_map(|x: uint| if x.is_even() { Some(x*x) } else { None });
        assert_eq!(it.to_vec(), ~[0*0, 2*2, 4*4, 6*6, 8*8]);
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

    #[test]
    fn test_iterator_nth() {
        let v = &[0, 1, 2, 3, 4];
        for uint::range(0, v.len()) |i| {
            assert_eq!(v.iter().nth(i).unwrap(), &v[i]);
        }
    }

    #[test]
    fn test_iterator_last() {
        let v = &[0, 1, 2, 3, 4];
        assert_eq!(v.iter().last().unwrap(), &4);
        assert_eq!(v.slice(0, 1).iter().last().unwrap(), &0);
    }

    #[test]
    fn test_iterator_count() {
        let v = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(v.slice(0, 4).iter().count(), 4);
        assert_eq!(v.slice(0, 10).iter().count(), 10);
        assert_eq!(v.slice(0, 0).iter().count(), 0);
    }

    #[test]
    fn test_iterator_sum() {
        let v = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(v.slice(0, 4).iter().transform(|&x| x).sum(), 6);
        assert_eq!(v.iter().transform(|&x| x).sum(), 55);
        assert_eq!(v.slice(0, 0).iter().transform(|&x| x).sum(), 0);
    }

    #[test]
    fn test_iterator_product() {
        let v = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(v.slice(0, 4).iter().transform(|&x| x).product(), 0);
        assert_eq!(v.slice(1, 5).iter().transform(|&x| x).product(), 24);
        assert_eq!(v.slice(0, 0).iter().transform(|&x| x).product(), 1);
    }

    #[test]
    fn test_iterator_max() {
        let v = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(v.slice(0, 4).iter().transform(|&x| x).max(), Some(3));
        assert_eq!(v.iter().transform(|&x| x).max(), Some(10));
        assert_eq!(v.slice(0, 0).iter().transform(|&x| x).max(), None);
    }

    #[test]
    fn test_iterator_min() {
        let v = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(v.slice(0, 4).iter().transform(|&x| x).min(), Some(0));
        assert_eq!(v.iter().transform(|&x| x).min(), Some(0));
        assert_eq!(v.slice(0, 0).iter().transform(|&x| x).min(), None);
    }

    #[test]
    fn test_all() {
        let v = ~&[1, 2, 3, 4, 5];
        assert!(v.iter().all(|&x| *x < 10));
        assert!(!v.iter().all(|&x| x.is_even()));
        assert!(!v.iter().all(|&x| *x > 100));
        assert!(v.slice(0, 0).iter().all(|_| fail!()));
    }

    #[test]
    fn test_any() {
        let v = ~&[1, 2, 3, 4, 5];
        assert!(v.iter().any(|&x| *x < 10));
        assert!(v.iter().any(|&x| x.is_even()));
        assert!(!v.iter().any(|&x| *x > 100));
        assert!(!v.slice(0, 0).iter().any(|_| fail!()));
    }
}
