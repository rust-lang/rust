// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Debug;
use std::iter::{self, FromIterator};
use std::slice;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Range};
use std::fmt;
use std::vec;
use std::u32;

use rustc_serialize as serialize;

/// Represents some newtyped `usize` wrapper.
///
/// (purpose: avoid mixing indexes for different bitvector domains.)
pub trait Idx: Copy + 'static + Eq + Debug {
    fn new(usize) -> Self;
    fn index(self) -> usize;
}

impl Idx for usize {
    fn new(idx: usize) -> Self { idx }
    fn index(self) -> usize { self }
}

impl Idx for u32 {
    fn new(idx: usize) -> Self { assert!(idx <= u32::MAX as usize); idx as u32 }
    fn index(self) -> usize { self as usize }
}

#[derive(Clone)]
pub struct IndexVec<I: Idx, T> {
    pub raw: Vec<T>,
    _marker: PhantomData<Fn(&I)>
}

impl<I: Idx, T: serialize::Encodable> serialize::Encodable for IndexVec<I, T> {
    fn encode<S: serialize::Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        serialize::Encodable::encode(&self.raw, s)
    }
}

impl<I: Idx, T: serialize::Decodable> serialize::Decodable for IndexVec<I, T> {
    fn decode<D: serialize::Decoder>(d: &mut D) -> Result<Self, D::Error> {
        serialize::Decodable::decode(d).map(|v| {
            IndexVec { raw: v, _marker: PhantomData }
        })
    }
}

impl<I: Idx, T: fmt::Debug> fmt::Debug for IndexVec<I, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.raw, fmt)
    }
}

pub type Enumerated<I, J> = iter::Map<iter::Enumerate<J>, IntoIdx<I>>;

impl<I: Idx, T> IndexVec<I, T> {
    #[inline]
    pub fn new() -> Self {
        IndexVec { raw: Vec::new(), _marker: PhantomData }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        IndexVec { raw: Vec::with_capacity(capacity), _marker: PhantomData }
    }

    #[inline]
    pub fn from_elem<S>(elem: T, universe: &IndexVec<I, S>) -> Self
        where T: Clone
    {
        IndexVec { raw: vec![elem; universe.len()], _marker: PhantomData }
    }

    #[inline]
    pub fn from_elem_n(elem: T, n: usize) -> Self
        where T: Clone
    {
        IndexVec { raw: vec![elem; n], _marker: PhantomData }
    }

    #[inline]
    pub fn push(&mut self, d: T) -> I {
        let idx = I::new(self.len());
        self.raw.push(d);
        idx
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    #[inline]
    pub fn into_iter(self) -> vec::IntoIter<T> {
        self.raw.into_iter()
    }

    #[inline]
    pub fn into_iter_enumerated(self) -> Enumerated<I, vec::IntoIter<T>>
    {
        self.raw.into_iter().enumerate().map(IntoIdx { _marker: PhantomData })
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<T> {
        self.raw.iter()
    }

    #[inline]
    pub fn iter_enumerated(&self) -> Enumerated<I, slice::Iter<T>>
    {
        self.raw.iter().enumerate().map(IntoIdx { _marker: PhantomData })
    }

    #[inline]
    pub fn indices(&self) -> iter::Map<Range<usize>, IntoIdx<I>> {
        (0..self.len()).map(IntoIdx { _marker: PhantomData })
    }

    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<T> {
        self.raw.iter_mut()
    }

    #[inline]
    pub fn iter_enumerated_mut(&mut self) -> Enumerated<I, slice::IterMut<T>>
    {
        self.raw.iter_mut().enumerate().map(IntoIdx { _marker: PhantomData })
    }

    #[inline]
    pub fn last(&self) -> Option<I> {
        self.len().checked_sub(1).map(I::new)
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.raw.shrink_to_fit()
    }

    #[inline]
    pub fn swap(&mut self, a: usize, b: usize) {
        self.raw.swap(a, b)
    }

    #[inline]
    pub fn truncate(&mut self, a: usize) {
        self.raw.truncate(a)
    }
}

impl<I: Idx, T> Index<I> for IndexVec<I, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: I) -> &T {
        &self.raw[index.index()]
    }
}

impl<I: Idx, T> IndexMut<I> for IndexVec<I, T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut T {
        &mut self.raw[index.index()]
    }
}

impl<I: Idx, T> Extend<T> for IndexVec<I, T> {
    #[inline]
    fn extend<J: IntoIterator<Item = T>>(&mut self, iter: J) {
        self.raw.extend(iter);
    }
}

impl<I: Idx, T> FromIterator<T> for IndexVec<I, T> {
    #[inline]
    fn from_iter<J>(iter: J) -> Self where J: IntoIterator<Item=T> {
        IndexVec { raw: FromIterator::from_iter(iter), _marker: PhantomData }
    }
}

impl<I: Idx, T> IntoIterator for IndexVec<I, T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    #[inline]
    fn into_iter(self) -> vec::IntoIter<T> {
        self.raw.into_iter()
    }

}

impl<'a, I: Idx, T> IntoIterator for &'a IndexVec<I, T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> slice::Iter<'a, T> {
        self.raw.iter()
    }
}

impl<'a, I: Idx, T> IntoIterator for &'a mut IndexVec<I, T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(mut self) -> slice::IterMut<'a, T> {
        self.raw.iter_mut()
    }
}

pub struct IntoIdx<I: Idx> { _marker: PhantomData<fn(&I)> }
impl<I: Idx, T> FnOnce<((usize, T),)> for IntoIdx<I> {
    type Output = (I, T);

    extern "rust-call" fn call_once(self, ((n, t),): ((usize, T),)) -> Self::Output {
        (I::new(n), t)
    }
}

impl<I: Idx, T> FnMut<((usize, T),)> for IntoIdx<I> {
    extern "rust-call" fn call_mut(&mut self, ((n, t),): ((usize, T),)) -> Self::Output {
        (I::new(n), t)
    }
}

impl<I: Idx> FnOnce<(usize,)> for IntoIdx<I> {
    type Output = I;

    extern "rust-call" fn call_once(self, (n,): (usize,)) -> Self::Output {
        I::new(n)
    }
}

impl<I: Idx> FnMut<(usize,)> for IntoIdx<I> {
    extern "rust-call" fn call_mut(&mut self, (n,): (usize,)) -> Self::Output {
        I::new(n)
    }
}
