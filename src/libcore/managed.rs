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
use rt::rt_fail_borrowed;

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

#[cfg(target_word_size = "32")]
const FROZEN_BIT: uint = 0x80000000;
#[cfg(target_word_size = "64")]
const FROZEN_BIT: uint = 0x8000000000000000;

#[inline(always)]
pub pure fn ptr_eq<T>(a: @T, b: @T) -> bool {
    //! Determine if two shared boxes point to the same object
    unsafe { ptr::addr_of(&(*a)) == ptr::addr_of(&(*b)) }
}

#[lang="borrow_as_imm"]
#[inline(always)]
pub unsafe fn borrow_as_imm(a: *u8) {
    let a: *mut BoxRepr = transmute(a);
    (*a).header.ref_count |= FROZEN_BIT;
}

#[lang="return_to_mut"]
#[inline(always)]
pub unsafe fn return_to_mut(a: *u8) {
    let a: *mut BoxRepr = transmute(a);
    (*a).header.ref_count &= !FROZEN_BIT;
}

#[lang="check_not_borrowed"]
#[inline(always)]
pub unsafe fn check_not_borrowed(a: *u8) {
    let a: *mut BoxRepr = transmute(a);
    if ((*a).header.ref_count & FROZEN_BIT) != 0 {
        rt_fail_borrowed();
    }
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
