// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Bit fiddling on positive IEEE 754 floats. Negative numbers aren't and
//! needn't be handled.
//! Normal floating point numbers have a canonical representation as (frac,
//! exp) such that the
//! value is 2^exp * (1 + sum(frac[N-i] / 2^i)) where N is the number of bits.
//! Subnormals are
//! slightly different and weird, but the same principle applies.
//!
//! Here, however, we represent them as (sig, k) with f positive, such that the
//! value is f * 2^e.
//! Besides making the "hidden bit" explicit, this changes the exponent by the
//! so-called
//! mantissa shift.
//!
//! Put another way, normally floats are written as (1) but here they are
//! written as (2):
//!
//! 1. `1.101100...11 * 2^m`
//! 2. `1101100...11 * 2^n`
//!
//! We call (1) the **fractional representation** and (2) the **integral
//! representation**.
//!
//! Many functions in this module only handle normal numbers. The dec2flt
//! routines conservatively
//! take the universally-correct slow path (Algorithm M) for very small and
//! very large numbers.
//! That algorithm needs only next_float() which does handle subnormals and
//! zeros.
use prelude::v1::*;
use u32;
use cmp::Ordering::{Less, Equal, Greater};
use ops::{Mul, Div, Neg};
use fmt::{Debug, LowerExp};
use mem::transmute;
use num::diy_float::Fp;
use num::FpCategory::{Infinite, Zero, Subnormal, Normal, Nan};
use num::Float;
use num::dec2flt::num::{self, Big};

#[derive(Copy, Clone, Debug)]
pub struct Unpacked {
    pub sig: u64,
    pub k: i16,
}

impl Unpacked {
    pub fn new(sig: u64, k: i16) -> Self {
        Unpacked { sig: sig, k: k }
    }
}

/// A helper trait to avoid duplicating basically all the conversion code for `f32` and `f64`.
///
/// See the parent module's doc comment for why this is necessary.
///
/// Should **never ever** be implemented for other types or be used outside the dec2flt module.
/// Inherits from `Float` because there is some overlap, but all the reused methods are trivial.
/// The "methods" (pseudo-constants) with default implementation should not be overriden.
pub trait RawFloat : Float + Copy + Debug + LowerExp
                    + Mul<Output=Self> + Div<Output=Self> + Neg<Output=Self>
{
    /// Get the raw binary representation of the float.
    fn transmute(self) -> u64;

    /// Transmute the raw binary representation into a float.
    fn from_bits(bits: u64) -> Self;

    /// Decode the float.
    fn unpack(self) -> Unpacked;

    /// Cast from a small integer that can be represented exactly.  Panic if the integer can't be
    /// represented, the other code in this module makes sure to never let that happen.
    fn from_int(x: u64) -> Self;

    // FIXME Everything that follows should be associated constants, but taking the
    // value of an
    // associated constant from a type parameter does not work (yet?)
    // A possible workaround is having a `FloatInfo` struct for all the constants,
    // but so far
    // the methods aren't painful enough to rewrite.

    /// What the name says. It's easier to hard code than juggling intrinsics and
    /// hoping LLVM constant folds it.
    fn ceil_log5_of_max_sig() -> i16;

    // A conservative bound on the decimal digits of inputs that can't produce
    // overflow or zero or
    /// subnormals. Probably the decimal exponent of the maximum normal value, hence the name.
    fn max_normal_digits() -> usize;

    /// When the most significant decimal digit has a place value greater than this, the number
    /// is certainly rounded to infinity.
    fn inf_cutoff() -> i64;

    /// When the most significant decimal digit has a place value less than this, the number
    /// is certainly rounded to zero.
    fn zero_cutoff() -> i64;

    /// The number of bits in the exponent.
    fn exp_bits() -> u8;

    /// The number of bits in the singificand, *including* the hidden bit.
    fn sig_bits() -> u8;

    /// The number of bits in the singificand, *excluding* the hidden bit.
    fn explicit_sig_bits() -> u8 {
        Self::sig_bits() - 1
    }

    /// The maximum legal exponent in fractional representation.
    fn max_exp() -> i16 {
        (1 << (Self::exp_bits() - 1)) - 1
    }

    /// The minimum legal exponent in fractional representation, excluding subnormals.
    fn min_exp() -> i16 {
        -Self::max_exp() + 1
    }

    /// `MAX_EXP` for integral representation, i.e., with the shift applied.
    fn max_exp_int() -> i16 {
        Self::max_exp() - (Self::sig_bits() as i16 - 1)
    }

    /// `MAX_EXP` encoded (i.e., with offset bias)
    fn max_encoded_exp() -> i16 {
        (1 << Self::exp_bits()) - 1
    }

    /// `MIN_EXP` for integral representation, i.e., with the shift applied.
    fn min_exp_int() -> i16 {
        Self::min_exp() - (Self::sig_bits() as i16 - 1)
    }

    /// The maximum normalized singificand in integral representation.
    fn max_sig() -> u64 {
        (1 << Self::sig_bits()) - 1
    }

    /// The minimal normalized significand in integral representation.
    fn min_sig() -> u64 {
        1 << (Self::sig_bits() - 1)
    }
}

impl RawFloat for f32 {
    fn sig_bits() -> u8 {
        24
    }

    fn exp_bits() -> u8 {
        8
    }

    fn ceil_log5_of_max_sig() -> i16 {
        11
    }

    fn transmute(self) -> u64 {
        let bits: u32 = unsafe { transmute(self) };
        bits as u64
    }

    fn from_bits(bits: u64) -> f32 {
        assert!(bits < u32::MAX as u64, "f32::from_bits: too many bits");
        unsafe { transmute(bits as u32) }
    }

    fn unpack(self) -> Unpacked {
        let (sig, exp, _sig) = self.integer_decode();
        Unpacked::new(sig, exp)
    }

    fn from_int(x: u64) -> f32 {
        // rkruppe is uncertain whether `as` rounds correctly on all platforms.
        debug_assert!(x as f32 == fp_to_float(Fp { f: x, e: 0 }));
        x as f32
    }

    fn max_normal_digits() -> usize {
        35
    }

    fn inf_cutoff() -> i64 {
        40
    }

    fn zero_cutoff() -> i64 {
        -48
    }
}


impl RawFloat for f64 {
    fn sig_bits() -> u8 {
        53
    }

    fn exp_bits() -> u8 {
        11
    }

    fn ceil_log5_of_max_sig() -> i16 {
        23
    }

    fn transmute(self) -> u64 {
        let bits: u64 = unsafe { transmute(self) };
        bits
    }

    fn from_bits(bits: u64) -> f64 {
        unsafe { transmute(bits) }
    }

    fn unpack(self) -> Unpacked {
        let (sig, exp, _sig) = self.integer_decode();
        Unpacked::new(sig, exp)
    }

    fn from_int(x: u64) -> f64 {
        // rkruppe is uncertain whether `as` rounds correctly on all platforms.
        debug_assert!(x as f64 == fp_to_float(Fp { f: x, e: 0 }));
        x as f64
    }

    fn max_normal_digits() -> usize {
        305
    }

    fn inf_cutoff() -> i64 {
        310
    }

    fn zero_cutoff() -> i64 {
        -326
    }

}

/// Convert an Fp to the closest f64. Only handles number that fit into a normalized f64.
pub fn fp_to_float<T: RawFloat>(x: Fp) -> T {
    let x = x.normalize();
    // x.f is 64 bit, so x.e has a mantissa shift of 63
    let e = x.e + 63;
    if e > T::max_exp() {
        panic!("fp_to_float: exponent {} too large", e)
    } else if e > T::min_exp() {
        encode_normal(round_normal::<T>(x))
    } else {
        panic!("fp_to_float: exponent {} too small", e)
    }
}

/// Round the 64-bit significand to 53 bit with half-to-even. Does not handle exponent overflow.
pub fn round_normal<T: RawFloat>(x: Fp) -> Unpacked {
    let excess = 64 - T::sig_bits() as i16;
    let half: u64 = 1 << (excess - 1);
    let (q, rem) = (x.f >> excess, x.f & ((1 << excess) - 1));
    assert_eq!(q << excess | rem, x.f);
    // Adjust mantissa shift
    let k = x.e + excess;
    if rem < half {
        Unpacked::new(q, k)
    } else if rem == half && (q % 2) == 0 {
        Unpacked::new(q, k)
    } else if q == T::max_sig() {
        Unpacked::new(T::min_sig(), k + 1)
    } else {
        Unpacked::new(q + 1, k)
    }
}

/// Inverse of `RawFloat::unpack()` for normalized numbers.
/// Panics if the significand or exponent are not valid for normalized numbers.
pub fn encode_normal<T: RawFloat>(x: Unpacked) -> T {
    debug_assert!(T::min_sig() <= x.sig && x.sig <= T::max_sig(),
                  "encode_normal: significand not normalized");
    // Remove the hidden bit
    let sig_enc = x.sig & !(1 << T::explicit_sig_bits());
    // Adjust the exponent for exponent bias and mantissa shift
    let k_enc = x.k + T::max_exp() + T::explicit_sig_bits() as i16;
    debug_assert!(k_enc != 0 && k_enc < T::max_encoded_exp(),
                  "encode_normal: exponent out of range");
    // Leave sign bit at 0 ("+"), our numbers are all positive
    let bits = (k_enc as u64) << T::explicit_sig_bits() | sig_enc;
    T::from_bits(bits)
}

/// Construct the subnormal. A mantissa of 0 is allowed and constructs zero.
pub fn encode_subnormal<T: RawFloat>(significand: u64) -> T {
    assert!(significand < T::min_sig(),
            "encode_subnormal: not actually subnormal");
    // Êncoded exponent is 0, the sign bit is 0, so we just have to reinterpret the
    // bits.
    T::from_bits(significand)
}

/// Approximate a bignum with an Fp. Rounds within 0.5 ULP with half-to-even.
pub fn big_to_fp(f: &Big) -> Fp {
    let end = f.bit_length();
    assert!(end != 0, "big_to_fp: unexpectedly, input is zero");
    let start = end.saturating_sub(64);
    let leading = num::get_bits(f, start, end);
    // We cut off all bits prior to the index `start`, i.e., we effectively
    // right-shift by
    // an amount of `start`, so this is also the exponent we need.
    let e = start as i16;
    let rounded_down = Fp { f: leading, e: e }.normalize();
    // Round (half-to-even) depending on the truncated bits.
    match num::compare_with_half_ulp(f, start) {
        Less => rounded_down,
        Equal if leading % 2 == 0 => rounded_down,
        Equal | Greater => match leading.checked_add(1) {
            Some(f) => Fp { f: f, e: e }.normalize(),
            None => Fp {
                f: 1 << 63,
                e: e + 1,
            },
        },
    }
}

/// Find the largest floating point number strictly smaller than the argument.
/// Does not handle subnormals, zero, or exponent underflow.
pub fn prev_float<T: RawFloat>(x: T) -> T {
    match x.classify() {
        Infinite => panic!("prev_float: argument is infinite"),
        Nan => panic!("prev_float: argument is NaN"),
        Subnormal => panic!("prev_float: argument is subnormal"),
        Zero => panic!("prev_float: argument is zero"),
        Normal => {
            let Unpacked { sig, k } = x.unpack();
            if sig == T::min_sig() {
                encode_normal(Unpacked::new(T::max_sig(), k - 1))
            } else {
                encode_normal(Unpacked::new(sig - 1, k))
            }
        }
    }
}

// Find the smallest floating point number strictly larger than the argument.
// This operation is saturating, i.e. next_float(inf) == inf.
// Unlike most code in this module, this function does handle zero, subnormals,
// and infinities.
// However, like all other code here, it does not deal with NaN and negative
// numbers.
pub fn next_float<T: RawFloat>(x: T) -> T {
    match x.classify() {
        Nan => panic!("next_float: argument is NaN"),
        Infinite => T::infinity(),
        // This seems too good to be true, but it works.
        // 0.0 is encoded as the all-zero word. Subnormals are 0x000m...m where m is the mantissa.
        // In particular, the smallest subnormal is 0x0...01 and the largest is 0x000F...F.
        // The smallest normal number is 0x0010...0, so this corner case works as well.
        // If the increment overflows the mantissa, the carry bit increments the exponent as we
        // want, and the mantissa bits become zero. Because of the hidden bit convention, this
        // too is exactly what we want!
        // Finally, f64::MAX + 1 = 7eff...f + 1 = 7ff0...0 = f64::INFINITY.
        Zero | Subnormal | Normal => {
            let bits: u64 = x.transmute();
            T::from_bits(bits + 1)
        }
    }
}
