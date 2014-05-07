// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for architecture-sized unsigned integers (`uint` type)

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

uint_module!(uint, int, ::int::BITS)

#[cfg(target_word_size = "32")]
impl CheckedAdd for uint {
    #[inline]
    fn checked_add(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u32_add_with_overflow(*self as u32, *v as u32);
            if y { None } else { Some(x as uint) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedAdd for uint {
    #[inline]
    fn checked_add(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u64_add_with_overflow(*self as u64, *v as u64);
            if y { None } else { Some(x as uint) }
        }
    }
}

#[cfg(target_word_size = "32")]
impl CheckedSub for uint {
    #[inline]
    fn checked_sub(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u32_sub_with_overflow(*self as u32, *v as u32);
            if y { None } else { Some(x as uint) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedSub for uint {
    #[inline]
    fn checked_sub(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u64_sub_with_overflow(*self as u64, *v as u64);
            if y { None } else { Some(x as uint) }
        }
    }
}

#[cfg(target_word_size = "32")]
impl CheckedMul for uint {
    #[inline]
    fn checked_mul(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u32_mul_with_overflow(*self as u32, *v as u32);
            if y { None } else { Some(x as uint) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedMul for uint {
    #[inline]
    fn checked_mul(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u64_mul_with_overflow(*self as u64, *v as u64);
            if y { None } else { Some(x as uint) }
        }
    }
}
