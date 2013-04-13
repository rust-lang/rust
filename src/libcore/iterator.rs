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

pub trait Iterator<T> {
    /// Advance the iterator and return the next value. Return `None` when the end is reached.
    fn next(&mut self) -> Option<T>;
}

/// A shim implementing the `for` loop iteration protocol for iterator objects
#[inline]
pub fn advance<T, U: Iterator<T>>(iter: &mut U, f: &fn(T) -> bool) {
    loop {
        match iter.next() {
            Some(x) => {
                if !f(x) { return }
            }
            None => return
        }
    }
}

pub struct ZipIterator<T, U> {
    priv a: T,
    priv b: U
}

pub impl<A, B, T: Iterator<A>, U: Iterator<B>> ZipIterator<T, U> {
    #[inline(always)]
    fn new(a: T, b: U) -> ZipIterator<T, U> {
        ZipIterator{a: a, b: b}
    }
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

pub impl<'self, A, T: Iterator<A>> FilterIterator<'self, A, T> {
    #[inline(always)]
    fn new(iter: T, predicate: &'self fn(&A) -> bool) -> FilterIterator<'self, A, T> {
        FilterIterator{iter: iter, predicate: predicate}
    }
}

impl<'self, A, T: Iterator<A>> Iterator<A> for FilterIterator<'self, A, T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        for advance(self) |x| {
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

pub impl<'self, A, B, T: Iterator<A>> MapIterator<'self, A, B, T> {
    #[inline(always)]
    fn new(iter: T, f: &'self fn(A) -> B) -> MapIterator<'self, A, B, T> {
        MapIterator{iter: iter, f: f}
    }
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
