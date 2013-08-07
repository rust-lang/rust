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

#[cfg(not(test))] use cmp::*;

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
