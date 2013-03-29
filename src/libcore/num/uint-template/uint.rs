// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for `uint`

use num::NumCast;

pub use self::inst::{
    div_ceil, div_round, div_floor, iterate,
    next_power_of_two
};

pub mod inst {
    use sys;
    use iter;

    pub type T = uint;
    #[allow(non_camel_case_types)]
    pub type T_SIGNED = int;

    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    #[cfg(target_arch = "mips")]
    pub static bits: uint = 32;

    #[cfg(target_arch = "x86_64")]
    pub static bits: uint = 64;

    /**
    * Divide two numbers, return the result, rounded up.
    *
    * # Arguments
    *
    * * x - an integer
    * * y - an integer distinct from 0u
    *
    * # Return value
    *
    * The smallest integer `q` such that `x/y <= q`.
    */
    pub fn div_ceil(x: uint, y: uint) -> uint {
        let div = x / y;
        if x % y == 0u { div }
        else { div + 1u }
    }

    /**
    * Divide two numbers, return the result, rounded to the closest integer.
    *
    * # Arguments
    *
    * * x - an integer
    * * y - an integer distinct from 0u
    *
    * # Return value
    *
    * The integer `q` closest to `x/y`.
    */
    pub fn div_round(x: uint, y: uint) -> uint {
        let div = x / y;
        if x % y * 2u  < y { div }
        else { div + 1u }
    }

    /**
    * Divide two numbers, return the result, rounded down.
    *
    * Note: This is the same function as `div`.
    *
    * # Arguments
    *
    * * x - an integer
    * * y - an integer distinct from 0u
    *
    * # Return value
    *
    * The smallest integer `q` such that `x/y <= q`. This
    * is either `x/y` or `x/y + 1`.
    */
    pub fn div_floor(x: uint, y: uint) -> uint { return x / y; }

    /**
    * Iterate over the range [`lo`..`hi`), or stop when requested
    *
    * # Arguments
    *
    * * lo - The integer at which to start the loop (included)
    * * hi - The integer at which to stop the loop (excluded)
    * * it - A block to execute with each consecutive integer of the range.
    *        Return `true` to continue, `false` to stop.
    *
    * # Return value
    *
    * `true` If execution proceeded correctly, `false` if it was interrupted,
    * that is if `it` returned `false` at any point.
    */
    pub fn iterate(lo: uint, hi: uint, it: &fn(uint) -> bool) -> bool {
        let mut i = lo;
        while i < hi {
            if (!it(i)) { return false; }
            i += 1u;
        }
        return true;
    }

    impl iter::Times for uint {
        #[inline(always)]
        /**
        * A convenience form for basic iteration. Given a uint `x`,
        * `for x.times { ... }` executes the given block x times.
        *
        * Equivalent to `for uint::range(0, x) |_| { ... }`.
        *
        * Not defined on all integer types to permit unambiguous
        * use with integer literals of inferred integer-type as
        * the self-value (eg. `for 100.times { ... }`).
        */
        fn times(&self, it: &fn() -> bool) {
            let mut i = *self;
            while i > 0 {
                if !it() { break }
                i -= 1;
            }
        }
    }

    /// Returns the smallest power of 2 greater than or equal to `n`
    #[inline(always)]
    pub fn next_power_of_two(n: uint) -> uint {
        let halfbits: uint = sys::size_of::<uint>() * 4u;
        let mut tmp: uint = n - 1u;
        let mut shift: uint = 1u;
        while shift <= halfbits { tmp |= tmp >> shift; shift <<= 1u; }
        return tmp + 1u;
    }

    #[test]
    fn test_next_power_of_two() {
        assert!((next_power_of_two(0u) == 0u));
        assert!((next_power_of_two(1u) == 1u));
        assert!((next_power_of_two(2u) == 2u));
        assert!((next_power_of_two(3u) == 4u));
        assert!((next_power_of_two(4u) == 4u));
        assert!((next_power_of_two(5u) == 8u));
        assert!((next_power_of_two(6u) == 8u));
        assert!((next_power_of_two(7u) == 8u));
        assert!((next_power_of_two(8u) == 8u));
        assert!((next_power_of_two(9u) == 16u));
        assert!((next_power_of_two(10u) == 16u));
        assert!((next_power_of_two(11u) == 16u));
        assert!((next_power_of_two(12u) == 16u));
        assert!((next_power_of_two(13u) == 16u));
        assert!((next_power_of_two(14u) == 16u));
        assert!((next_power_of_two(15u) == 16u));
        assert!((next_power_of_two(16u) == 16u));
        assert!((next_power_of_two(17u) == 32u));
        assert!((next_power_of_two(18u) == 32u));
        assert!((next_power_of_two(19u) == 32u));
        assert!((next_power_of_two(20u) == 32u));
        assert!((next_power_of_two(21u) == 32u));
        assert!((next_power_of_two(22u) == 32u));
        assert!((next_power_of_two(23u) == 32u));
        assert!((next_power_of_two(24u) == 32u));
        assert!((next_power_of_two(25u) == 32u));
        assert!((next_power_of_two(26u) == 32u));
        assert!((next_power_of_two(27u) == 32u));
        assert!((next_power_of_two(28u) == 32u));
        assert!((next_power_of_two(29u) == 32u));
        assert!((next_power_of_two(30u) == 32u));
        assert!((next_power_of_two(31u) == 32u));
        assert!((next_power_of_two(32u) == 32u));
        assert!((next_power_of_two(33u) == 64u));
        assert!((next_power_of_two(34u) == 64u));
        assert!((next_power_of_two(35u) == 64u));
        assert!((next_power_of_two(36u) == 64u));
        assert!((next_power_of_two(37u) == 64u));
        assert!((next_power_of_two(38u) == 64u));
        assert!((next_power_of_two(39u) == 64u));
    }

    #[test]
    fn test_overflows() {
        use uint;
        assert!((uint::max_value > 0u));
        assert!((uint::min_value <= 0u));
        assert!((uint::min_value + uint::max_value + 1u == 0u));
    }

    #[test]
    fn test_div() {
        assert!((div_floor(3u, 4u) == 0u));
        assert!((div_ceil(3u, 4u)  == 1u));
        assert!((div_round(3u, 4u) == 1u));
    }

    #[test]
    pub fn test_times() {
        use iter::Times;
        let ten = 10 as uint;
        let mut accum = 0;
        for ten.times { accum += 1; }
        assert!((accum == 10));
    }
}

impl NumCast for uint {
    /**
     * Cast `n` to a `uint`
     */
    #[inline(always)]
    fn from<N:NumCast>(n: N) -> uint { n.to_uint() }

    #[inline(always)] fn to_u8(&self)    -> u8    { *self as u8    }
    #[inline(always)] fn to_u16(&self)   -> u16   { *self as u16   }
    #[inline(always)] fn to_u32(&self)   -> u32   { *self as u32   }
    #[inline(always)] fn to_u64(&self)   -> u64   { *self as u64   }
    #[inline(always)] fn to_uint(&self)  -> uint  { *self          }

    #[inline(always)] fn to_i8(&self)    -> i8    { *self as i8    }
    #[inline(always)] fn to_i16(&self)   -> i16   { *self as i16   }
    #[inline(always)] fn to_i32(&self)   -> i32   { *self as i32   }
    #[inline(always)] fn to_i64(&self)   -> i64   { *self as i64   }
    #[inline(always)] fn to_int(&self)   -> int   { *self as int   }

    #[inline(always)] fn to_f32(&self)   -> f32   { *self as f32   }
    #[inline(always)] fn to_f64(&self)   -> f64   { *self as f64   }
    #[inline(always)] fn to_float(&self) -> float { *self as float }
}

#[test]
fn test_numcast() {
    assert!((20u   == 20u.to_uint()));
    assert!((20u8  == 20u.to_u8()));
    assert!((20u16 == 20u.to_u16()));
    assert!((20u32 == 20u.to_u32()));
    assert!((20u64 == 20u.to_u64()));
    assert!((20i   == 20u.to_int()));
    assert!((20i8  == 20u.to_i8()));
    assert!((20i16 == 20u.to_i16()));
    assert!((20i32 == 20u.to_i32()));
    assert!((20i64 == 20u.to_i64()));
    assert!((20f   == 20u.to_float()));
    assert!((20f32 == 20u.to_f32()));
    assert!((20f64 == 20u.to_f64()));

    assert!((20u == NumCast::from(20u)));
    assert!((20u == NumCast::from(20u8)));
    assert!((20u == NumCast::from(20u16)));
    assert!((20u == NumCast::from(20u32)));
    assert!((20u == NumCast::from(20u64)));
    assert!((20u == NumCast::from(20i)));
    assert!((20u == NumCast::from(20i8)));
    assert!((20u == NumCast::from(20i16)));
    assert!((20u == NumCast::from(20i32)));
    assert!((20u == NumCast::from(20i64)));
    assert!((20u == NumCast::from(20f)));
    assert!((20u == NumCast::from(20f32)));
    assert!((20u == NumCast::from(20f64)));

    assert!((20u == num::cast(20u)));
    assert!((20u == num::cast(20u8)));
    assert!((20u == num::cast(20u16)));
    assert!((20u == num::cast(20u32)));
    assert!((20u == num::cast(20u64)));
    assert!((20u == num::cast(20i)));
    assert!((20u == num::cast(20i8)));
    assert!((20u == num::cast(20i16)));
    assert!((20u == num::cast(20i32)));
    assert!((20u == num::cast(20i64)));
    assert!((20u == num::cast(20f)));
    assert!((20u == num::cast(20f32)));
    assert!((20u == num::cast(20f64)));
}
