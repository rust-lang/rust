// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Functions for the unit type.

*/

#[cfg(not(test))]
use prelude::*;

#[cfg(not(test))]
impl Eq for () {
    #[inline(always)]
    fn eq(&self, _other: &()) -> bool { true }
    #[inline(always)]
    fn ne(&self, _other: &()) -> bool { false }
}

#[cfg(not(test))]
impl Ord for () {
    #[inline(always)]
    fn lt(&self, _other: &()) -> bool { false }
    #[inline(always)]
    fn le(&self, _other: &()) -> bool { true }
    #[inline(always)]
    fn ge(&self, _other: &()) -> bool { true }
    #[inline(always)]
    fn gt(&self, _other: &()) -> bool { false }
}

#[cfg(not(test))]
impl TotalOrd for () {
    #[inline(always)]
    fn cmp(&self, _other: &()) -> Ordering { Equal }
}

#[cfg(not(test))]
impl TotalEq for () {
    #[inline(always)]
    fn equals(&self, _other: &()) -> bool { true }
}
