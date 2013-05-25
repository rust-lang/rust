// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for `i16`

use num::BitCount;
use unstable::intrinsics;

pub use self::generated::*;

int_module!(i16, 16)

impl BitCount for i16 {
    /// Counts the number of bits set. Wraps LLVM's `ctpop` intrinsic.
    #[inline(always)]
    fn population_count(&self) -> i16 { unsafe { intrinsics::ctpop16(*self) } }

    /// Counts the number of leading zeros. Wraps LLVM's `ctlz` intrinsic.
    #[inline(always)]
    fn leading_zeros(&self) -> i16 { unsafe { intrinsics::ctlz16(*self) } }

    /// Counts the number of trailing zeros. Wraps LLVM's `cttz` intrinsic.
    #[inline(always)]
    fn trailing_zeros(&self) -> i16 { unsafe { intrinsics::cttz16(*self) } }
}
