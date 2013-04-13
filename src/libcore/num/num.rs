// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An interface for numeric types
use cmp::{Eq, Ord};
use ops::{Neg, Add, Sub, Mul, Div, Modulo};
use option::Option;
use kinds::Copy;

pub mod strconv;

pub trait IntConvertible {
    fn to_int(&self) -> int;
    fn from_int(n: int) -> Self;
}

pub trait Zero {
    fn zero() -> Self;
}

pub trait One {
    fn one() -> Self;
}

pub fn abs<T:Ord + Zero + Neg<T>>(v: T) -> T {
    if v < Zero::zero() { v.neg() } else { v }
}

pub trait Round {
    fn round(&self, mode: RoundMode) -> Self;

    fn floor(&self) -> Self;
    fn ceil(&self)  -> Self;
    fn fract(&self) -> Self;
}

pub enum RoundMode {
    RoundDown,
    RoundUp,
    RoundToZero,
    RoundFromZero
}

/**
 * Cast from one machine scalar to another
 *
 * # Example
 *
 * ~~~
 * let twenty: f32 = num::cast(0x14);
 * assert!(twenty == 20f32);
 * ~~~
 */
#[inline(always)]
pub fn cast<T:NumCast,U:NumCast>(n: T) -> U {
    NumCast::from(n)
}

/**
 * An interface for casting between machine scalars
 */
pub trait NumCast {
    fn from<T:NumCast>(n: T) -> Self;

    fn to_u8(&self) -> u8;
    fn to_u16(&self) -> u16;
    fn to_u32(&self) -> u32;
    fn to_u64(&self) -> u64;
    fn to_uint(&self) -> uint;

    fn to_i8(&self) -> i8;
    fn to_i16(&self) -> i16;
    fn to_i32(&self) -> i32;
    fn to_i64(&self) -> i64;
    fn to_int(&self) -> int;

    fn to_f32(&self) -> f32;
    fn to_f64(&self) -> f64;
    fn to_float(&self) -> float;
}

macro_rules! impl_num_cast(
    ($T:ty, $conv:ident) => (
        // FIXME #4375: This enclosing module is necessary because
        // of a bug with macros expanding into multiple items 
        pub mod $conv {
            use num::NumCast;
            
            #[cfg(notest)]
            impl NumCast for $T {
                #[doc = "Cast `n` to a `$T`"]
                #[inline(always)]
                fn from<N:NumCast>(n: N) -> $T {
                    // `$conv` could be generated using `concat_idents!`, but that
                    // macro seems to be broken at the moment
                    n.$conv()
                }

                #[inline(always)] fn to_u8(&self)    -> u8    { *self as u8    }
                #[inline(always)] fn to_u16(&self)   -> u16   { *self as u16   }
                #[inline(always)] fn to_u32(&self)   -> u32   { *self as u32   }
                #[inline(always)] fn to_u64(&self)   -> u64   { *self as u64   }
                #[inline(always)] fn to_uint(&self)  -> uint  { *self as uint  }

                #[inline(always)] fn to_i8(&self)    -> i8    { *self as i8    }
                #[inline(always)] fn to_i16(&self)   -> i16   { *self as i16   }
                #[inline(always)] fn to_i32(&self)   -> i32   { *self as i32   }
                #[inline(always)] fn to_i64(&self)   -> i64   { *self as i64   }
                #[inline(always)] fn to_int(&self)   -> int   { *self as int   }

                #[inline(always)] fn to_f32(&self)   -> f32   { *self as f32   }
                #[inline(always)] fn to_f64(&self)   -> f64   { *self as f64   }
                #[inline(always)] fn to_float(&self) -> float { *self as float }
            }
        }
    )
)

impl_num_cast!(u8,    to_u8)
impl_num_cast!(u16,   to_u16)
impl_num_cast!(u32,   to_u32)
impl_num_cast!(u64,   to_u64)
impl_num_cast!(uint,  to_uint)
impl_num_cast!(i8,    to_i8)
impl_num_cast!(i16,   to_i16)
impl_num_cast!(i32,   to_i32)
impl_num_cast!(i64,   to_i64)
impl_num_cast!(int,   to_int)
impl_num_cast!(f32,   to_f32)
impl_num_cast!(f64,   to_f64)
impl_num_cast!(float, to_float)

pub trait ToStrRadix {
    pub fn to_str_radix(&self, radix: uint) -> ~str;
}

pub trait FromStrRadix {
    pub fn from_str_radix(str: &str, radix: uint) -> Option<Self>;
}

// Generic math functions:

/**
 * Calculates a power to a given radix, optimized for uint `pow` and `radix`.
 *
 * Returns `radix^pow` as `T`.
 *
 * Note:
 * Also returns `1` for `0^0`, despite that technically being an
 * undefined number. The reason for this is twofold:
 * - If code written to use this function cares about that special case, it's
 *   probably going to catch it before making the call.
 * - If code written to use this function doesn't care about it, it's
 *   probably assuming that `x^0` always equals `1`.
 */
pub fn pow_with_uint<T:NumCast+One+Zero+Copy+Div<T,T>+Mul<T,T>>(
    radix: uint, pow: uint) -> T {
    let _0: T = Zero::zero();
    let _1: T = One::one();

    if pow   == 0u { return _1; }
    if radix == 0u { return _0; }
    let mut my_pow     = pow;
    let mut total      = _1;
    let mut multiplier = cast(radix as int);
    while (my_pow > 0u) {
        if my_pow % 2u == 1u {
            total *= multiplier;
        }
        my_pow     /= 2u;
        multiplier *= multiplier;
    }
    total
}

macro_rules! test_cast_20(
    ($_20:expr) => ({
        let _20 = $_20;
        
        assert!(20u   == _20.to_uint());
        assert!(20u8  == _20.to_u8());
        assert!(20u16 == _20.to_u16());
        assert!(20u32 == _20.to_u32());
        assert!(20u64 == _20.to_u64());
        assert!(20i   == _20.to_int());
        assert!(20i8  == _20.to_i8());
        assert!(20i16 == _20.to_i16());
        assert!(20i32 == _20.to_i32());
        assert!(20i64 == _20.to_i64());
        assert!(20f   == _20.to_float());
        assert!(20f32 == _20.to_f32());
        assert!(20f64 == _20.to_f64());

        assert!(_20 == NumCast::from(20u));
        assert!(_20 == NumCast::from(20u8));
        assert!(_20 == NumCast::from(20u16));
        assert!(_20 == NumCast::from(20u32));
        assert!(_20 == NumCast::from(20u64));
        assert!(_20 == NumCast::from(20i));
        assert!(_20 == NumCast::from(20i8));
        assert!(_20 == NumCast::from(20i16));
        assert!(_20 == NumCast::from(20i32));
        assert!(_20 == NumCast::from(20i64));
        assert!(_20 == NumCast::from(20f));
        assert!(_20 == NumCast::from(20f32));
        assert!(_20 == NumCast::from(20f64));

        assert!(_20 == cast(20u));
        assert!(_20 == cast(20u8));
        assert!(_20 == cast(20u16));
        assert!(_20 == cast(20u32));
        assert!(_20 == cast(20u64));
        assert!(_20 == cast(20i));
        assert!(_20 == cast(20i8));
        assert!(_20 == cast(20i16));
        assert!(_20 == cast(20i32));
        assert!(_20 == cast(20i64));
        assert!(_20 == cast(20f));
        assert!(_20 == cast(20f32));
        assert!(_20 == cast(20f64));
    })
)

#[test] fn test_u8_cast()    { test_cast_20!(20u8)  }
#[test] fn test_u16_cast()   { test_cast_20!(20u16) }
#[test] fn test_u32_cast()   { test_cast_20!(20u32) }
#[test] fn test_u64_cast()   { test_cast_20!(20u64) }
#[test] fn test_uint_cast()  { test_cast_20!(20u)   }
#[test] fn test_i8_cast()    { test_cast_20!(20i8)  }
#[test] fn test_i16_cast()   { test_cast_20!(20i16) }
#[test] fn test_i32_cast()   { test_cast_20!(20i32) }
#[test] fn test_i64_cast()   { test_cast_20!(20i64) }
#[test] fn test_int_cast()   { test_cast_20!(20i)   }
#[test] fn test_f32_cast()   { test_cast_20!(20f32) }
#[test] fn test_f64_cast()   { test_cast_20!(20f64) }
#[test] fn test_float_cast() { test_cast_20!(20f)   }

#[test]
fn test_generic_cast() {
    use ops::Add;
    
    fn add_2<T: Add<T,T> + NumCast>(n: T) -> T {
        n + cast(2)
    }
    
    assert!(add_2(1u)   == 3u);
    assert!(add_2(1u8)  == 3u8);
    assert!(add_2(1u16) == 3u16);
    assert!(add_2(1u32) == 3u32);
    assert!(add_2(1u64) == 3u64);
    assert!(add_2(1i)   == 3i);
    assert!(add_2(1i8)  == 3i8);
    assert!(add_2(1i16) == 3i16);
    assert!(add_2(1i32) == 3i32);
    assert!(add_2(1i64) == 3i64);
    assert!(add_2(1f)   == 3f);
    assert!(add_2(1f32) == 3f32);
    assert!(add_2(1f64) == 3f64);
}
