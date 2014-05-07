// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for unsigned 8-bits integers (`u8` type)

use default::Default;
use intrinsics;
use num::{Bitwise, Bounded, Zero, One, Unsigned, Num, Int, Primitive};
use num::{CheckedAdd, CheckedSub, CheckedMul, CheckedDiv};
use option::{Some, None, Option};

#[cfg(not(test))]
use cmp::{Eq, Ord, TotalEq, TotalOrd, Less, Greater, Equal, Ordering};
#[cfg(not(test))]
use ops::{Add, Sub, Mul, Div, Rem, Neg, BitAnd, BitOr, BitXor};
#[cfg(not(test))]
use ops::{Shl, Shr, Not};

uint_module!(u8, i8, 8)

impl CheckedAdd for u8 {
    #[inline]
    fn checked_add(&self, v: &u8) -> Option<u8> {
        unsafe {
            let (x, y) = intrinsics::u8_add_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedSub for u8 {
    #[inline]
    fn checked_sub(&self, v: &u8) -> Option<u8> {
        unsafe {
            let (x, y) = intrinsics::u8_sub_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedMul for u8 {
    #[inline]
    fn checked_mul(&self, v: &u8) -> Option<u8> {
        unsafe {
            let (x, y) = intrinsics::u8_mul_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}
