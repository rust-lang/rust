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
//! used to store the values on the heap. This type does not support re-allocating onto the heap,
//! and there is no way to push more elements onto the existing storage.
//!
//! The N above is determined by Array's implementor, by way of an associatated constant.

use std::ops::Deref;
use std::iter::{IntoIterator, FromIterator};

use array_vec::{Array, ArrayVec};

#[derive(Debug)]
pub enum AccumulateVec<A: Array> {
    Array(ArrayVec<A>),
    Heap(Vec<A::Element>)
}

impl<A: Array> Deref for AccumulateVec<A> {
    type Target = [A::Element];
    fn deref(&self) -> &Self::Target {
        match *self {
            AccumulateVec::Array(ref v) => &v[..],
            AccumulateVec::Heap(ref v) => &v[..],
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

