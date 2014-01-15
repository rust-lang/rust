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

use std::vec;

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
            Empty => {
                *self = Vec(~[t]);
            }
        }
    }

    pub fn pop(&mut self) -> T {
        match *self {
            Vec(ref mut v) => v.pop(),
            Empty => fail!("pop from empty opt_vec")
        }
    }

    pub fn last<'a>(&'a self) -> &'a T {
        match *self {
            Vec(ref v) => v.last(),
            Empty => fail!("last on empty opt_vec")
        }
    }

    pub fn mut_last<'a>(&'a mut self) -> &'a mut T {
        match *self {
            Vec(ref mut v) => v.mut_last(),
            Empty => fail!("mut_last on empty opt_vec")
        }
    }

    pub fn map<U>(&self, op: |&T| -> U) -> OptVec<U> {
        match *self {
            Empty => Empty,
            Vec(ref v) => Vec(v.map(op))
        }
    }

    pub fn map_move<U>(self, op: |T| -> U) -> OptVec<U> {
        match self {
            Empty => Empty,
            Vec(v) => Vec(v.move_iter().map(op).collect())
        }
    }

    pub fn get<'a>(&'a self, i: uint) -> &'a T {
        match *self {
            Empty => fail!("Invalid index {}", i),
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

    pub fn swap_remove(&mut self, index: uint) {
        match *self {
            Empty => { fail!("Index out of bounds"); }
            Vec(ref mut v) => {
                assert!(index < v.len());
                v.swap_remove(index);
            }
        }
    }

    #[inline]
    pub fn iter<'r>(&'r self) -> Items<'r, T> {
        match *self {
            Empty => Items{iter: None},
            Vec(ref v) => Items{iter: Some(v.iter())}
        }
    }

    #[inline]
    pub fn map_to_vec<B>(&self, op: |&T| -> B) -> ~[B] {
        self.iter().map(op).collect()
    }

    pub fn mapi_to_vec<B>(&self, op: |uint, &T| -> B) -> ~[B] {
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

pub struct Items<'a, T> {
    priv iter: Option<vec::Items<'a, T>>
}

impl<'a, T> Iterator<&'a T> for Items<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<&'a T> {
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

impl<'a, T> DoubleEndedIterator<&'a T> for Items<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        match self.iter {
            Some(ref mut x) => x.next_back(),
            None => None
        }
    }
}

impl<A> FromIterator<A> for OptVec<A> {
    fn from_iterator<T: Iterator<A>>(iterator: &mut T) -> OptVec<A> {
        let mut r = Empty;
        for x in *iterator {
            r.push(x);
        }
        r
    }
}
