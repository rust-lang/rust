// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Functions for the unit type.

#[cfg(not(test))]
use default::Default;
#[cfg(not(test))]
use cmp::{Eq, Equal, Ord, Ordering, TotalEq, TotalOrd};
use fmt;

#[cfg(not(test))]
impl Eq for () {
    #[inline]
    fn eq(&self, _other: &()) -> bool { true }
    #[inline]
    fn ne(&self, _other: &()) -> bool { false }
}

#[cfg(not(test))]
impl Ord for () {
    #[inline]
    fn lt(&self, _other: &()) -> bool { false }
}

#[cfg(not(test))]
impl TotalOrd for () {
    #[inline]
    fn cmp(&self, _other: &()) -> Ordering { Equal }
}

#[cfg(not(test))]
impl TotalEq for () {}

#[cfg(not(test))]
impl Default for () {
    #[inline]
    fn default() -> () { () }
}

impl fmt::Show for () {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("()")
    }
}
