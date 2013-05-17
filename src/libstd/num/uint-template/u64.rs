// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for `u64`

mod inst {
    use num::{Primitive, BitCount};
    use unstable::intrinsics;

    pub type T = u64;
    #[allow(non_camel_case_types)]
    pub type T_SIGNED = i64;
    pub static bits: uint = 64;

    impl Primitive for u64 {
        #[inline(always)]
        fn bits() -> uint { 64 }

        #[inline(always)]
        fn bytes() -> uint { Primitive::bits::<u64>() / 8 }
    }

    impl BitCount for u64 {
        /// Counts the number of bits set. Wraps LLVM's `ctpop` intrinsic.
        #[inline(always)]
        fn population_count(&self) -> u64 { unsafe { intrinsics::ctpop64(*self as i64) as u64 } }

        /// Counts the number of leading zeros. Wraps LLVM's `ctlz` intrinsic.
        #[inline(always)]
        fn leading_zeros(&self) -> u64 { unsafe { intrinsics::ctlz64(*self as i64) as u64 } }

        /// Counts the number of trailing zeros. Wraps LLVM's `cttz` intrinsic.
        #[inline(always)]
        fn trailing_zeros(&self) -> u64 { unsafe { intrinsics::cttz64(*self as i64) as u64 } }
    }
}
