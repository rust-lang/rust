// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for signed 8-bits integers (`i8` type)

#![allow(non_uppercase_statics)]

use prelude::*;

use default::Default;
use from_str::FromStr;
use num::{Bitwise, Bounded, CheckedAdd, CheckedSub, CheckedMul};
use num::{CheckedDiv, Zero, One, strconv};
use num::{ToStrRadix, FromStrRadix};
use option::{Option, Some, None};
use str;
use intrinsics;

int_module!(i8, 8)

impl Bitwise for i8 {
    /// Returns the number of ones in the binary representation of the number.
    #[inline]
    fn count_ones(&self) -> i8 { unsafe { intrinsics::ctpop8(*self) } }

    /// Returns the number of leading zeros in the in the binary representation
    /// of the number.
    #[inline]
    fn leading_zeros(&self) -> i8 { unsafe { intrinsics::ctlz8(*self) } }

    /// Returns the number of trailing zeros in the in the binary representation
    /// of the number.
    #[inline]
    fn trailing_zeros(&self) -> i8 { unsafe { intrinsics::cttz8(*self) } }
}

impl CheckedAdd for i8 {
    #[inline]
    fn checked_add(&self, v: &i8) -> Option<i8> {
        unsafe {
            let (x, y) = intrinsics::i8_add_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedSub for i8 {
    #[inline]
    fn checked_sub(&self, v: &i8) -> Option<i8> {
        unsafe {
            let (x, y) = intrinsics::i8_sub_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedMul for i8 {
    #[inline]
    fn checked_mul(&self, v: &i8) -> Option<i8> {
        unsafe {
            let (x, y) = intrinsics::i8_mul_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}
