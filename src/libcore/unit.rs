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

Functions for the unit type.

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::{Eq, Ord};

#[cfg(notest)]
impl () : Eq {
    pure fn eq(&self, _other: &()) -> bool { true }
    pure fn ne(&self, _other: &()) -> bool { false }
}

#[cfg(notest)]
impl () : Ord {
    pure fn lt(&self, _other: &()) -> bool { false }
    pure fn le(&self, _other: &()) -> bool { true }
    pure fn ge(&self, _other: &()) -> bool { true }
    pure fn gt(&self, _other: &()) -> bool { false }
}

