// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for `u32`

mod inst {
    use num::{Primitive, BitCount};
    use unstable::intrinsics;

    pub type T = u32;
    #[allow(non_camel_case_types)]
    pub type T_SIGNED = i32;
    pub static bits: uint = 32;

    impl Primitive for u32 {
        #[inline(always)]
        fn bits() -> uint { 32 }

        #[inline(always)]
        fn bytes() -> uint { Primitive::bits::<u32>() / 8 }
    }

    impl BitCount for u32 {
        /// Counts the number of bits set. Wraps LLVM's `ctpop` intrinsic.
        #[inline(always)]
        fn population_count(&self) -> u32 { unsafe { intrinsics::ctpop32(*self as i32) as u32 } }

        /// Counts the number of leading zeros. Wraps LLVM's `ctlp` intrinsic.
        #[inline(always)]
        fn leading_zeros(&self) -> u32 { unsafe { intrinsics::ctlz32(*self as i32) as u32 } }

        /// Counts the number of trailing zeros. Wraps LLVM's `cttp` intrinsic.
        #[inline(always)]
        fn trailing_zeros(&self) -> u32 { unsafe { intrinsics::cttz32(*self as i32) as u32 } }
    }
}
