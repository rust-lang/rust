// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations on tuples

use clone::Clone;
use kinds::Copy;
use vec;

#[cfg(notest)] use cmp::{Eq, Ord};

pub trait CopyableTuple<T, U> {
    fn first(&self) -> T;
    fn second(&self) -> U;
    fn swap(&self) -> (U, T);
}

impl<T:Copy,U:Copy> CopyableTuple<T, U> for (T, U) {

    /// Return the first element of self
    #[inline(always)]
    fn first(&self) -> T {
        let (t, _) = *self;
        return t;
    }

    /// Return the second element of self
    #[inline(always)]
    fn second(&self) -> U {
        let (_, u) = *self;
        return u;
    }

    /// Return the results of swapping the two elements of self
    #[inline(always)]
    fn swap(&self) -> (U, T) {
        let (t, u) = *self;
        return (u, t);
    }

}

impl<T:Clone,U:Clone> Clone for (T, U) {
    fn clone(&self) -> (T, U) {
        let (a, b) = match *self {
            (ref a, ref b) => (a, b)
        };
        (a.clone(), b.clone())
    }
}

pub trait ImmutableTuple<T, U> {
    fn first_ref(&self) -> &'self T;
    fn second_ref(&self) -> &'self U;
}

impl<T, U> ImmutableTuple<T, U> for (T, U) {
    #[inline(always)]
    fn first_ref(&self) -> &'self T {
        match *self {
            (ref t, _) => t,
        }
    }
    #[inline(always)]
    fn second_ref(&self) -> &'self U {
        match *self {
            (_, ref u) => u,
        }
    }
}

pub trait ExtendedTupleOps<A,B> {
    fn zip(&self) -> ~[(A, B)];
    fn map<C>(&self, f: &fn(a: &A, b: &B) -> C) -> ~[C];
}

impl<'self,A:Copy,B:Copy> ExtendedTupleOps<A,B> for (&'self [A], &'self [B]) {
    #[inline(always)]
    fn zip(&self) -> ~[(A, B)] {
        match *self {
            (ref a, ref b) => {
                vec::zip_slice(*a, *b)
            }
        }
    }

    #[inline(always)]
    fn map<C>(&self, f: &fn(a: &A, b: &B) -> C) -> ~[C] {
        match *self {
            (ref a, ref b) => {
                vec::map2(*a, *b, f)
            }
        }
    }
}

impl<A:Copy,B:Copy> ExtendedTupleOps<A,B> for (~[A], ~[B]) {

    #[inline(always)]
    fn zip(&self) -> ~[(A, B)] {
        match *self {
            (ref a, ref b) => {
                vec::zip_slice(*a, *b)
            }
        }
    }

    #[inline(always)]
    fn map<C>(&self, f: &fn(a: &A, b: &B) -> C) -> ~[C] {
        match *self {
            (ref a, ref b) => {
                vec::map2(*a, *b, f)
            }
        }
    }
}

#[cfg(notest)]
impl<A:Eq> Eq for (A,) {
    #[inline(always)]
    fn eq(&self, other: &(A,)) -> bool {
        match (*self) {
            (ref self_a,) => match other {
                &(ref other_a,) => {
                    (*self_a).eq(other_a)
                }
            }
        }
    }
    #[inline(always)]
    fn ne(&self, other: &(A,)) -> bool { !(*self).eq(other) }
}

#[cfg(notest)]
impl<A:Ord> Ord for (A,) {
    #[inline(always)]
    fn lt(&self, other: &(A,)) -> bool {
        match (*self) {
            (ref self_a,) => {
                match (*other) {
                    (ref other_a,) => {
                        if (*self_a).lt(other_a) { return true; }
                        return false;
                    }
                }
            }
        }
    }
    #[inline(always)]
    fn le(&self, other: &(A,)) -> bool { !other.lt(&(*self)) }
    #[inline(always)]
    fn ge(&self, other: &(A,)) -> bool { !self.lt(other) }
    #[inline(always)]
    fn gt(&self, other: &(A,)) -> bool { other.lt(&(*self))  }
}


#[cfg(notest)]
impl<A:Eq,B:Eq> Eq for (A, B) {
    #[inline(always)]
    fn eq(&self, other: &(A, B)) -> bool {
        match (*self) {
            (ref self_a, ref self_b) => match other {
                &(ref other_a, ref other_b) => {
                    (*self_a).eq(other_a) && (*self_b).eq(other_b)
                }
            }
        }
    }
    #[inline(always)]
    fn ne(&self, other: &(A, B)) -> bool { !(*self).eq(other) }
}

#[cfg(notest)]
impl<A:Ord,B:Ord> Ord for (A, B) {
    #[inline(always)]
    fn lt(&self, other: &(A, B)) -> bool {
        match (*self) {
            (ref self_a, ref self_b) => {
                match (*other) {
                    (ref other_a, ref other_b) => {
                        if (*self_a).lt(other_a) { return true; }
                        if (*other_a).lt(self_a) { return false; }
                        if (*self_b).lt(other_b) { return true; }
                        return false;
                    }
                }
            }
        }
    }
    #[inline(always)]
    fn le(&self, other: &(A, B)) -> bool { !(*other).lt(&(*self)) }
    #[inline(always)]
    fn ge(&self, other: &(A, B)) -> bool { !(*self).lt(other) }
    #[inline(always)]
    fn gt(&self, other: &(A, B)) -> bool { (*other).lt(&(*self))  }
}

#[cfg(notest)]
impl<A:Eq,B:Eq,C:Eq> Eq for (A, B, C) {
    #[inline(always)]
    fn eq(&self, other: &(A, B, C)) -> bool {
        match (*self) {
            (ref self_a, ref self_b, ref self_c) => match other {
                &(ref other_a, ref other_b, ref other_c) => {
                    (*self_a).eq(other_a) && (*self_b).eq(other_b)
                        && (*self_c).eq(other_c)
                }
            }
        }
    }
    #[inline(always)]
    fn ne(&self, other: &(A, B, C)) -> bool { !(*self).eq(other) }
}

#[cfg(notest)]
impl<A:Ord,B:Ord,C:Ord> Ord for (A, B, C) {
    #[inline(always)]
    fn lt(&self, other: &(A, B, C)) -> bool {
        match (*self) {
            (ref self_a, ref self_b, ref self_c) => {
                match (*other) {
                    (ref other_a, ref other_b, ref other_c) => {
                        if (*self_a).lt(other_a) { return true; }
                        if (*other_a).lt(self_a) { return false; }
                        if (*self_b).lt(other_b) { return true; }
                        if (*other_b).lt(self_b) { return false; }
                        if (*self_c).lt(other_c) { return true; }
                        return false;
                    }
                }
            }
        }
    }
    #[inline(always)]
    fn le(&self, other: &(A, B, C)) -> bool { !(*other).lt(&(*self)) }
    #[inline(always)]
    fn ge(&self, other: &(A, B, C)) -> bool { !(*self).lt(other) }
    #[inline(always)]
    fn gt(&self, other: &(A, B, C)) -> bool { (*other).lt(&(*self))  }
}

#[test]
fn test_tuple_ref() {
    let x = (~"foo", ~"bar");
    assert!(x.first_ref() == &~"foo");
    assert!(x.second_ref() == &~"bar");
}

#[test]
#[allow(non_implicitly_copyable_typarams)]
fn test_tuple() {
    assert!((948, 4039.48).first() == 948);
    assert!((34.5, ~"foo").second() == ~"foo");
    assert!(('a', 2).swap() == (2, 'a'));
}

#[test]
fn test_clone() {
    let a = (1, ~"2");
    let b = a.clone();
    assert!(a.first() == b.first());
    assert!(a.second() == b.second());
}
