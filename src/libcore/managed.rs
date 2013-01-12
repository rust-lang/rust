// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations on managed box types

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cast::transmute;
use cmp::{Eq, Ord};
use managed::raw::BoxRepr;
use prelude::*;
use ptr;

pub mod raw {
    use intrinsic::TyDesc;

    pub struct BoxHeaderRepr {
        ref_count: uint,
        type_desc: *TyDesc,
        prev: *BoxRepr,
        next: *BoxRepr,
    }

    pub struct BoxRepr {
        header: BoxHeaderRepr,
        data: u8
    }

}

#[inline(always)]
pub pure fn ptr_eq<T>(a: @T, b: @T) -> bool {
    //! Determine if two shared boxes point to the same object
    unsafe { ptr::addr_of(&(*a)) == ptr::addr_of(&(*b)) }
}

#[cfg(notest)]
impl<T:Eq> @const T : Eq {
    #[inline(always)]
    pure fn eq(&self, other: &@const T) -> bool { *(*self) == *(*other) }
    #[inline(always)]
    pure fn ne(&self, other: &@const T) -> bool { *(*self) != *(*other) }
}

#[cfg(notest)]
impl<T:Ord> @const T : Ord {
    #[inline(always)]
    pure fn lt(&self, other: &@const T) -> bool { *(*self) < *(*other) }
    #[inline(always)]
    pure fn le(&self, other: &@const T) -> bool { *(*self) <= *(*other) }
    #[inline(always)]
    pure fn ge(&self, other: &@const T) -> bool { *(*self) >= *(*other) }
    #[inline(always)]
    pure fn gt(&self, other: &@const T) -> bool { *(*self) > *(*other) }
}

#[test]
fn test() {
    let x = @3;
    let y = @3;
    assert (ptr_eq::<int>(x, x));
    assert (ptr_eq::<int>(y, y));
    assert (!ptr_eq::<int>(x, y));
    assert (!ptr_eq::<int>(y, x));
}
