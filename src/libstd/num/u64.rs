// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for unsigned 64-bits integer (`u64` type)

#[allow(non_uppercase_statics)];

use prelude::*;

use default::Default;
use num::{Bitwise, Bounded};
#[cfg(target_word_size = "64")]
use num::CheckedMul;
use num::{CheckedAdd, CheckedSub};
use num::{CheckedDiv, Zero, One, strconv};
use num::{ToStrRadix, FromStrRadix};
use option::{Option, Some, None};
use str;
use intrinsics;

uint_module!(u64, i64, 64)

impl CheckedAdd for u64 {
    #[inline]
    fn checked_add(&self, v: &u64) -> Option<u64> {
        unsafe {
            let (x, y) = intrinsics::u64_add_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedSub for u64 {
    #[inline]
    fn checked_sub(&self, v: &u64) -> Option<u64> {
        unsafe {
            let (x, y) = intrinsics::u64_sub_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}

impl CheckedMul for u64 {
    #[inline]
    fn checked_mul(&self, v: &u64) -> Option<u64> {
        unsafe {
            let (x, y) = intrinsics::u64_mul_with_overflow(*self, *v);
            if y { None } else { Some(x) }
        }
    }
}
