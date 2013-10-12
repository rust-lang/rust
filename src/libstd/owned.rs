// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations on unique pointer types

#[allow(missing_doc)];
#[cfg(not(test))] use cmp::*;
use either::{Either, Right};

#[cfg(not(test))]
impl<T:Eq> Eq for ~T {
    #[inline]
    fn eq(&self, other: &~T) -> bool { *(*self) == *(*other) }
    #[inline]
    fn ne(&self, other: &~T) -> bool { *(*self) != *(*other) }
}

#[cfg(not(test))]
impl<T:Ord> Ord for ~T {
    #[inline]
    fn lt(&self, other: &~T) -> bool { *(*self) < *(*other) }
    #[inline]
    fn le(&self, other: &~T) -> bool { *(*self) <= *(*other) }
    #[inline]
    fn ge(&self, other: &~T) -> bool { *(*self) >= *(*other) }
    #[inline]
    fn gt(&self, other: &~T) -> bool { *(*self) > *(*other) }
}

#[cfg(not(test))]
impl<T: TotalOrd> TotalOrd for ~T {
    #[inline]
    fn cmp(&self, other: &~T) -> Ordering { (**self).cmp(*other) }
}

#[cfg(not(test))]
impl<T: TotalEq> TotalEq for ~T {
    #[inline]
    fn equals(&self, other: &~T) -> bool { (**self).equals(*other) }
}

#[deriving(Clone)]
pub struct Own<T> {
    priv p: ~T
}

impl<T> Own<T> {
    pub fn new(v: T) -> Own<T> {Own {p: ~v}}

    pub fn get<'r>(&'r self) -> &'r T {&*self.p}

    pub fn get_mut<'r>(&'r mut self) -> &'r mut T {&mut *self.p}
    pub fn unwrap(self) -> T {*self.p}

    pub fn try_get_mut<'r>(&'r mut self) -> Either<&'r mut Own<T>, &'r mut T> {
    Right(&mut *self.p)
    }
    pub fn try_unwrap<'r>(self) -> Either<Own<T>, T> {Right(*self.p)}

    pub fn cow<'r>(&'r mut self) -> &'r mut T {&mut *self.p}
    pub fn value<'r>(self) -> T {*self.p}
}
