// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for signed 16-bits integers (`i16` type)

#[allow(non_uppercase_statics)];

use prelude::*;

use default::Default;
use num::{BitCount, CheckedAdd, CheckedSub, CheckedMul};
use num::{CheckedDiv, Zero, One, strconv};
use num::{ToStrRadix, FromStrRadix};
use option::{Option, Some, None};
use str;
use unstable::intrinsics;

int_module!(i16, 16)

impl BitCount for i16 {
    /// Counts the number of bits set. Wraps LLVM's `ctpop` intrinsic.
    #[inline]
    fn population_count(&self) -> i16 { unsafe { intrinsics::ctpop16(*self) } }

    /// Counts the number of leading zeros. Wraps LLVM's `ctlz` intrinsic.
    #[inline]
    fn leading_zeros(&self) -> i16 { unsafe { intrinsics::ctlz16(*self) } }

    /// Counts the number of trailing zeros. Wraps LLVM's `cttz` intrinsic.
    #[inline]
    fn trailing_zeros(&self) -> i16 { unsafe { intrinsics::cttz16(*self) } }
}

impl CheckedAdd for i16 {
    #[inline]
    fn checked_add(&self, v: &i16) -> Option<i16> {
        unsafe {
            let (x, y) = intrinsics::i16_add_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedSub for i16 {
    #[inline]
    fn checked_sub(&self, v: &i16) -> Option<i16> {
        unsafe {
            let (x, y) = intrinsics::i16_sub_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedMul for i16 {
    #[inline]
    fn checked_mul(&self, v: &i16) -> Option<i16> {
        unsafe {
            let (x, y) = intrinsics::i16_mul_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}
