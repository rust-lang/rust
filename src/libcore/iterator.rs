// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Composable iterator objects

use prelude::*;

pub trait Iterator<A> {
    /// Advance the iterator and return the next value. Return `None` when the end is reached.
    fn next(&mut self) -> Option<A>;
}

pub trait IteratorUtil<A> {
    fn zip<B, U: Iterator<B>>(self, other: U) -> ZipIterator<Self, U>;
    // FIXME: #5898: should be called map
    fn transform<'r, B>(self, f: &'r fn(A) -> B) -> MapIterator<'r, A, B, Self>;
    fn filter<'r>(self, predicate: &'r fn(&A) -> bool) -> FilterIterator<'r, A, Self>;
    fn enumerate(self) -> EnumerateIterator<Self>;
    fn advance(&mut self, f: &fn(A) -> bool);
}

impl<A, T: Iterator<A>> IteratorUtil<A> for T {
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
