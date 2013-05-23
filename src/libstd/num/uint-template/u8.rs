// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for `u8`

mod inst {
    use num::{Primitive, BitCount};
    use unstable::intrinsics;

    pub type T = u8;
    #[allow(non_camel_case_types)]
    pub type T_SIGNED = i8;
    pub static bits: uint = 8;

    impl Primitive for u8 {
        #[inline(always)]
        fn bits() -> uint { 8 }

        #[inline(always)]
        fn bytes() -> uint { Primitive::bits::<u8>() / 8 }
    }

    impl BitCount for u8 {
        /// Counts the number of bits set. Wraps LLVM's `ctpop` intrinsic.
        #[inline(always)]
        fn population_count(&self) -> u8 { unsafe { intrinsics::ctpop8(*self as i8) as u8 } }

        /// Counts the number of leading zeros. Wraps LLVM's `ctlz` intrinsic.
        #[inline(always)]
        fn leading_zeros(&self) -> u8 { unsafe { intrinsics::ctlz8(*self as i8) as u8 } }

        /// Counts the number of trailing zeros. Wraps LLVM's `cttz` intrinsic.
        #[inline(always)]
        fn trailing_zeros(&self) -> u8 { unsafe { intrinsics::cttz8(*self as i8) as u8 } }
    }
}
