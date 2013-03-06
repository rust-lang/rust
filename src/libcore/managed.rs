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

use ptr;

#[cfg(notest)] use cmp::{Eq, Ord};

pub mod raw {

    pub const RC_EXCHANGE_UNIQUE : uint = (-1) as uint;
    pub const RC_MANAGED_UNIQUE : uint = (-2) as uint;
    pub const RC_IMMORTAL : uint = 0x77777777;

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

#[inline(always)]
pub pure fn mut_ptr_eq<T>(a: @mut T, b: @mut T) -> bool {
    //! Determine if two mutable shared boxes point to the same object
    unsafe { ptr::addr_of(&(*a)) == ptr::addr_of(&(*b)) }
}

#[cfg(notest)]
impl<T:Eq> Eq for @const T {
    #[inline(always)]
    pure fn eq(&self, other: &@const T) -> bool { *(*self) == *(*other) }
    #[inline(always)]
    pure fn ne(&self, other: &@const T) -> bool { *(*self) != *(*other) }
}

#[cfg(notest)]
impl<T:Ord> Ord for @const T {
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
    fail_unless!((ptr_eq::<int>(x, x)));
    fail_unless!((ptr_eq::<int>(y, y)));
    fail_unless!((!ptr_eq::<int>(x, y)));
    fail_unless!((!ptr_eq::<int>(y, x)));
}
