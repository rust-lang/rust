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

#![allow(non_uppercase_statics)]
#![allow(unsigned_negate)]

use prelude::*;

use default::Default;
use from_str::FromStr;
use num::{Bitwise, Bounded};
use num::{CheckedAdd, CheckedSub, CheckedMul};
use num::{CheckedDiv, Zero, One, strconv};
use num::{ToStrRadix, FromStrRadix};
use option::{Option, Some, None};
use str;
use intrinsics;

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
