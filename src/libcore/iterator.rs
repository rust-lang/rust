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

pub trait Iterator<A> {
    /// Advance the iterator and return the next value. Return `None` when the end is reached.
    fn next(&mut self) -> Option<A>;
}

/// Iterator adaptors provided for every `Iterator` implementation. The adaptor objects are also
/// implementations of the `Iterator` trait.
///
/// In the future these will be default methods instead of a utility trait.
pub trait IteratorUtil<A> {
    fn chain(self, other: Self) -> ChainIterator<Self>;
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
}

/// Iterator adaptors provided for every `Iterator` implementation. The adaptor objects are also
/// implementations of the `Iterator` trait.
///
/// In the future these will be default methods instead of a utility trait.
impl<A, T: Iterator<A>> IteratorUtil<A> for T {
    #[inline(always)]
    fn chain(self, other: T) -> ChainIterator<T> {
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
}

pub struct ChainIterator<T> {
    priv a: T,
    priv b: T,
    priv flag: bool
}

impl<A, T: Iterator<A>> Iterator<A> for ChainIterator<T> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use prelude::*;

    #[test]
    fn test_iterator_chain() {
        let xs = [0u, 1, 2, 3, 4, 5];
        let ys = [30, 40, 50, 60];
        let expected = [0, 1, 2, 3, 4, 5, 30, 40, 50, 60];
        let mut it = xs.iter().chain(ys.iter());
        let mut i = 0;
        for it.advance |&x: &uint| {
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
