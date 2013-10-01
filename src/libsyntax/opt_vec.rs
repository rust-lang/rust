// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Defines a type OptVec<T> that can be used in place of ~[T].
 * OptVec avoids the need for allocation for empty vectors.
 * OptVec implements the iterable interface as well as
 * other useful things like `push()` and `len()`.
 */

use std::vec::{VecIterator};

#[deriving(Clone, Encodable, Decodable, IterBytes)]
pub enum OptVec<T> {
    Empty,
    Vec(~[T])
}

pub fn with<T>(t: T) -> OptVec<T> {
    Vec(~[t])
}

pub fn from<T>(t: ~[T]) -> OptVec<T> {
    if t.len() == 0 {
        Empty
    } else {
        Vec(t)
    }
}

impl<T> OptVec<T> {
    pub fn push(&mut self, t: T) {
        match *self {
            Vec(ref mut v) => {
                v.push(t);
                return;
            }
            Empty => {}
        }

        // FIXME(#5074): flow insensitive means we can't move
        // assignment inside `match`
        *self = Vec(~[t]);
    }

    pub fn map<U>(&self, op: &fn(&T) -> U) -> OptVec<U> {
        match *self {
            Empty => Empty,
            Vec(ref v) => Vec(v.map(op))
        }
    }

    pub fn map_move<U>(self, op: &fn(T) -> U) -> OptVec<U> {
        match self {
            Empty => Empty,
            Vec(v) => Vec(v.move_iter().map(op).collect())
        }
    }

    pub fn get<'a>(&'a self, i: uint) -> &'a T {
        match *self {
            Empty => fail2!("Invalid index {}", i),
            Vec(ref v) => &v[i]
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> uint {
        match *self {
            Empty => 0,
            Vec(ref v) => v.len()
        }
    }

    #[inline]
    pub fn iter<'r>(&'r self) -> OptVecIterator<'r, T> {
        match *self {
            Empty => OptVecIterator{iter: None},
            Vec(ref v) => OptVecIterator{iter: Some(v.iter())}
        }
    }

    #[inline]
    pub fn map_to_vec<B>(&self, op: &fn(&T) -> B) -> ~[B] {
        self.iter().map(op).collect()
    }

    pub fn mapi_to_vec<B>(&self, op: &fn(uint, &T) -> B) -> ~[B] {
        let mut index = 0;
        self.map_to_vec(|a| {
            let i = index;
            index += 1;
            op(i, a)
        })
    }
}

pub fn take_vec<T>(v: OptVec<T>) -> ~[T] {
    match v {
        Empty => ~[],
        Vec(v) => v
    }
}

impl<T:Clone> OptVec<T> {
    pub fn prepend(&self, t: T) -> OptVec<T> {
        let mut v0 = ~[t];
        match *self {
            Empty => {}
            Vec(ref v1) => { v0.push_all(*v1); }
        }
        return Vec(v0);
    }
}

impl<A:Eq> Eq for OptVec<A> {
    fn eq(&self, other: &OptVec<A>) -> bool {
        // Note: cannot use #[deriving(Eq)] here because
        // (Empty, Vec(~[])) ought to be equal.
        match (self, other) {
            (&Empty, &Empty) => true,
            (&Empty, &Vec(ref v)) => v.is_empty(),
            (&Vec(ref v), &Empty) => v.is_empty(),
            (&Vec(ref v1), &Vec(ref v2)) => *v1 == *v2
        }
    }

    fn ne(&self, other: &OptVec<A>) -> bool {
        !self.eq(other)
    }
}

impl<T> Default for OptVec<T> {
    fn default() -> OptVec<T> { Empty }
}

pub struct OptVecIterator<'self, T> {
    priv iter: Option<VecIterator<'self, T>>
}

impl<'self, T> Iterator<&'self T> for OptVecIterator<'self, T> {
    #[inline]
    fn next(&mut self) -> Option<&'self T> {
        match self.iter {
            Some(ref mut x) => x.next(),
            None => None
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        match self.iter {
            Some(ref x) => x.size_hint(),
            None => (0, Some(0))
        }
    }
}
