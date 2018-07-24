// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A vector type intended to be used for collecting from iterators onto the stack.
//!
//! Space for up to N elements is provided on the stack.  If more elements are collected, Vec is
//! used to store the values on the heap.
//!
//! The N above is determined by Array's implementor, by way of an associated constant.

use std::ops::{Deref, DerefMut, RangeBounds};
use std::iter::{self, IntoIterator, FromIterator};
use std::slice;
use std::vec;

use rustc_serialize::{Encodable, Encoder, Decodable, Decoder};

use array_vec::{self, Array, ArrayVec};

#[derive(Hash, Debug)]
pub enum AccumulateVec<A: Array> {
    Array(ArrayVec<A>),
    Heap(Vec<A::Element>)
}

impl<A> Clone for AccumulateVec<A>
    where A: Array,
          A::Element: Clone {
    fn clone(&self) -> Self {
        match *self {
            AccumulateVec::Array(ref arr) => AccumulateVec::Array(arr.clone()),
            AccumulateVec::Heap(ref vec) => AccumulateVec::Heap(vec.clone()),
        }
    }
}

impl<A: Array> AccumulateVec<A> {
    pub fn new() -> AccumulateVec<A> {
        AccumulateVec::Array(ArrayVec::new())
    }

    pub fn is_array(&self) -> bool {
        match self {
            AccumulateVec::Array(..) => true,
            AccumulateVec::Heap(..) => false,
        }
    }

    pub fn one(el: A::Element) -> Self {
        iter::once(el).collect()
    }

    pub fn many<I: IntoIterator<Item=A::Element>>(iter: I) -> Self {
        iter.into_iter().collect()
    }

    pub fn len(&self) -> usize {
        match *self {
            AccumulateVec::Array(ref arr) => arr.len(),
            AccumulateVec::Heap(ref vec) => vec.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn pop(&mut self) -> Option<A::Element> {
        match *self {
            AccumulateVec::Array(ref mut arr) => arr.pop(),
            AccumulateVec::Heap(ref mut vec) => vec.pop(),
        }
    }

    pub fn drain<R>(&mut self, range: R) -> Drain<A>
        where R: RangeBounds<usize>
    {
        match *self {
            AccumulateVec::Array(ref mut v) => {
                Drain::Array(v.drain(range))
            },
            AccumulateVec::Heap(ref mut v) => {
                Drain::Heap(v.drain(range))
            },
        }
    }
}

impl<A: Array> Deref for AccumulateVec<A> {
    type Target = [A::Element];
    fn deref(&self) -> &Self::Target {
        match *self {
            AccumulateVec::Array(ref v) => v,
            AccumulateVec::Heap(ref v) => v,
        }
    }
}

impl<A: Array> DerefMut for AccumulateVec<A> {
    fn deref_mut(&mut self) -> &mut [A::Element] {
        match *self {
            AccumulateVec::Array(ref mut v) => v,
            AccumulateVec::Heap(ref mut v) => v,
        }
    }
}

impl<A: Array> FromIterator<A::Element> for AccumulateVec<A> {
    fn from_iter<I>(iter: I) -> AccumulateVec<A> where I: IntoIterator<Item=A::Element> {
        let iter = iter.into_iter();
        if iter.size_hint().1.map_or(false, |n| n <= A::LEN) {
            let mut v = ArrayVec::new();
            v.extend(iter);
            AccumulateVec::Array(v)
        } else {
            AccumulateVec::Heap(iter.collect())
        }
    }
}

pub struct IntoIter<A: Array> {
    repr: IntoIterRepr<A>,
}

enum IntoIterRepr<A: Array> {
    Array(array_vec::Iter<A>),
    Heap(vec::IntoIter<A::Element>),
}

impl<A: Array> Iterator for IntoIter<A> {
    type Item = A::Element;

    fn next(&mut self) -> Option<A::Element> {
        match self.repr {
            IntoIterRepr::Array(ref mut arr) => arr.next(),
            IntoIterRepr::Heap(ref mut iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.repr {
            IntoIterRepr::Array(ref iter) => iter.size_hint(),
            IntoIterRepr::Heap(ref iter) => iter.size_hint(),
        }
    }
}

pub enum Drain<'a, A: Array>
        where A::Element: 'a
{
    Array(array_vec::Drain<'a, A>),
    Heap(vec::Drain<'a, A::Element>),
}

impl<'a, A: Array> Iterator for Drain<'a, A> {
    type Item = A::Element;

    fn next(&mut self) -> Option<A::Element> {
        match *self {
            Drain::Array(ref mut drain) => drain.next(),
            Drain::Heap(ref mut drain) => drain.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match *self {
            Drain::Array(ref drain) => drain.size_hint(),
            Drain::Heap(ref drain) => drain.size_hint(),
        }
    }
}

impl<A: Array> IntoIterator for AccumulateVec<A> {
    type Item = A::Element;
    type IntoIter = IntoIter<A>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            repr: match self {
                AccumulateVec::Array(arr) => IntoIterRepr::Array(arr.into_iter()),
                AccumulateVec::Heap(vec) => IntoIterRepr::Heap(vec.into_iter()),
            }
        }
    }
}

impl<'a, A: Array> IntoIterator for &'a AccumulateVec<A> {
    type Item = &'a A::Element;
    type IntoIter = slice::Iter<'a, A::Element>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, A: Array> IntoIterator for &'a mut AccumulateVec<A> {
    type Item = &'a mut A::Element;
    type IntoIter = slice::IterMut<'a, A::Element>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<A: Array> From<Vec<A::Element>> for AccumulateVec<A> {
    fn from(v: Vec<A::Element>) -> AccumulateVec<A> {
        AccumulateVec::many(v)
    }
}

impl<A: Array> Default for AccumulateVec<A> {
    fn default() -> AccumulateVec<A> {
        AccumulateVec::new()
    }
}

impl<A> Encodable for AccumulateVec<A>
    where A: Array,
          A::Element: Encodable {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s))?;
            }
            Ok(())
        })
    }
}

impl<A> Decodable for AccumulateVec<A>
    where A: Array,
          A::Element: Decodable {
    fn decode<D: Decoder>(d: &mut D) -> Result<AccumulateVec<A>, D::Error> {
        d.read_seq(|d, len| {
            (0..len).map(|i| d.read_seq_elt(i, |d| Decodable::decode(d))).collect()
        })
    }
}
