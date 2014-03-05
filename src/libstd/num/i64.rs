// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for signed 64-bits integers (`i64` type)

#[allow(non_uppercase_statics)];

use prelude::*;

use default::Default;
use from_str::FromStr;
#[cfg(target_word_size = "64")]
use num::CheckedMul;
use num::{Bitwise, Bounded, CheckedAdd, CheckedSub};
use num::{CheckedDiv, Zero, One, strconv};
use num::{ToStrRadix, FromStrRadix};
use option::{Option, Some, None};
use str;
use intrinsics;

int_module!(i64, 64)

impl Bitwise for i64 {
    /// Returns the number of ones in the binary representation of the number.
    #[inline]
    fn count_ones(&self) -> i64 { unsafe { intrinsics::ctpop64(*self) } }

    /// Returns the number of leading zeros in the in the binary representation
    /// of the number.
    #[inline]
    fn leading_zeros(&self) -> i64 { unsafe { intrinsics::ctlz64(*self) } }

    /// Counts the number of trailing zeros.
    #[inline]
    fn trailing_zeros(&self) -> i64 { unsafe { intrinsics::cttz64(*self) } }
}

impl CheckedAdd for i64 {
    #[inline]
    fn checked_add(&self, v: &i64) -> Option<i64> {
        unsafe {
            let (x, y) = intrinsics::i64_add_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedSub for i64 {
    #[inline]
    fn checked_sub(&self, v: &i64) -> Option<i64> {
        unsafe {
            let (x, y) = intrinsics::i64_sub_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedMul for i64 {
    #[inline]
    fn checked_mul(&self, v: &i64) -> Option<i64> {
        unsafe {
            let (x, y) = intrinsics::i64_mul_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}
