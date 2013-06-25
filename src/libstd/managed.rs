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

use ptr::to_unsafe_ptr;

#[cfg(not(test))] use cmp::{Eq, Ord};

pub mod raw {
    use intrinsic::TyDesc;

    pub static RC_EXCHANGE_UNIQUE : uint = (-1) as uint;
    pub static RC_MANAGED_UNIQUE : uint = (-2) as uint;
    pub static RC_IMMORTAL : uint = 0x77777777;

    #[allow(missing_doc)]
    pub struct BoxHeaderRepr {
        ref_count: uint,
        type_desc: *TyDesc,
        prev: *BoxRepr,
        next: *BoxRepr,
    }

    #[allow(missing_doc)]
    pub struct BoxRepr {
        header: BoxHeaderRepr,
        data: u8
    }

}

/// Determine if two shared boxes point to the same object
#[inline]
pub fn ptr_eq<T>(a: @T, b: @T) -> bool {
    let (a_ptr, b_ptr): (*T, *T) = (to_unsafe_ptr(&*a), to_unsafe_ptr(&*b));
    a_ptr == b_ptr
}

/// Determine if two mutable shared boxes point to the same object
#[inline]
pub fn mut_ptr_eq<T>(a: @mut T, b: @mut T) -> bool {
    let (a_ptr, b_ptr): (*T, *T) = (to_unsafe_ptr(&*a), to_unsafe_ptr(&*b));
    a_ptr == b_ptr
}

#[cfg(not(test))]
impl<T:Eq> Eq for @T {
    #[inline]
    fn eq(&self, other: &@T) -> bool { *(*self) == *(*other) }
    #[inline]
    fn ne(&self, other: &@T) -> bool { *(*self) != *(*other) }
}

#[cfg(not(test))]
impl<T:Eq> Eq for @mut T {
    #[inline]
    fn eq(&self, other: &@mut T) -> bool { *(*self) == *(*other) }
    #[inline]
    fn ne(&self, other: &@mut T) -> bool { *(*self) != *(*other) }
}

#[cfg(not(test))]
impl<T:Ord> Ord for @T {
    #[inline]
    fn lt(&self, other: &@T) -> bool { *(*self) < *(*other) }
    #[inline]
    fn le(&self, other: &@T) -> bool { *(*self) <= *(*other) }
    #[inline]
    fn ge(&self, other: &@T) -> bool { *(*self) >= *(*other) }
    #[inline]
    fn gt(&self, other: &@T) -> bool { *(*self) > *(*other) }
}

#[cfg(not(test))]
impl<T:Ord> Ord for @mut T {
    #[inline]
    fn lt(&self, other: &@mut T) -> bool { *(*self) < *(*other) }
    #[inline]
    fn le(&self, other: &@mut T) -> bool { *(*self) <= *(*other) }
    #[inline]
    fn ge(&self, other: &@mut T) -> bool { *(*self) >= *(*other) }
    #[inline]
    fn gt(&self, other: &@mut T) -> bool { *(*self) > *(*other) }
}

#[test]
fn test() {
    let x = @3;
    let y = @3;
    assert!((ptr_eq::<int>(x, x)));
    assert!((ptr_eq::<int>(y, y)));
    assert!((!ptr_eq::<int>(x, y)));
    assert!((!ptr_eq::<int>(y, x)));
}
