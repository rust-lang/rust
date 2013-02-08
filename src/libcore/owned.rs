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

use cmp::{Eq, Ord};

#[cfg(notest)]
impl<T:Eq> ~const T : Eq {
    #[inline(always)]
    pure fn eq(&self, other: &~const T) -> bool { *(*self) == *(*other) }
    #[inline(always)]
    pure fn ne(&self, other: &~const T) -> bool { *(*self) != *(*other) }
}

#[cfg(notest)]
impl<T:Ord> ~const T : Ord {
    #[inline(always)]
    pure fn lt(&self, other: &~const T) -> bool { *(*self) < *(*other) }
    #[inline(always)]
    pure fn le(&self, other: &~const T) -> bool { *(*self) <= *(*other) }
    #[inline(always)]
    pure fn ge(&self, other: &~const T) -> bool { *(*self) >= *(*other) }
    #[inline(always)]
    pure fn gt(&self, other: &~const T) -> bool { *(*self) > *(*other) }
}

