// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for `int`

pub use self::inst::pow;

mod inst {
    pub type T = int;
    pub static bits: uint = ::uint::bits;

    /// Returns `base` raised to the power of `exponent`
    pub fn pow(base: int, exponent: uint) -> int {
        if exponent == 0u {
            //Not mathemtically true if ~[base == 0]
            return 1;
        }
        if base == 0 { return 0; }
        let mut my_pow  = exponent;
        let mut acc     = 1;
        let mut multiplier = base;
        while(my_pow > 0u) {
            if my_pow % 2u == 1u {
                acc *= multiplier;
            }
            my_pow     /= 2u;
            multiplier *= multiplier;
        }
        return acc;
    }

    #[test]
    fn test_pow() {
        assert!((pow(0, 0u) == 1));
        assert!((pow(0, 1u) == 0));
        assert!((pow(0, 2u) == 0));
        assert!((pow(-1, 0u) == 1));
        assert!((pow(1, 0u) == 1));
        assert!((pow(-3, 2u) == 9));
        assert!((pow(-3, 3u) == -27));
        assert!((pow(4, 9u) == 262144));
    }

    #[test]
    fn test_overflows() {
        assert!((::int::max_value > 0));
        assert!((::int::min_value <= 0));
        assert!((::int::min_value + ::int::max_value + 1 == 0));
    }
}
