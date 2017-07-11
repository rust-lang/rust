// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use {Category, ExpInt};
use {Float, FloatConvert, ParseError, Round, StatusAnd};

use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Neg;

#[must_use]
pub struct IeeeFloat<S> {
    marker: PhantomData<S>,
}

/// Represents floating point arithmetic semantics.
pub trait Semantics: Sized {
    /// Total number of bits in the in-memory format.
    const BITS: usize;

    /// Number of bits in the significand. This includes the integer bit.
    const PRECISION: usize;

    /// The largest E such that 2^E is representable; this matches the
    /// definition of IEEE 754.
    const MAX_EXP: ExpInt;

    /// The smallest E such that 2^E is a normalized number; this
    /// matches the definition of IEEE 754.
    const MIN_EXP: ExpInt = -Self::MAX_EXP + 1;
}

impl<S> Copy for IeeeFloat<S> {}
impl<S> Clone for IeeeFloat<S> {
    fn clone(&self) -> Self {
        *self
    }
}

macro_rules! ieee_semantics {
    ($($name:ident = $sem:ident($bits:tt : $exp_bits:tt)),*) => {
        $(pub struct $sem;)*
        $(pub type $name = IeeeFloat<$sem>;)*
        $(impl Semantics for $sem {
            const BITS: usize = $bits;
            const PRECISION: usize = ($bits - 1 - $exp_bits) + 1;
            const MAX_EXP: ExpInt = (1 << ($exp_bits - 1)) - 1;
        })*
    }
}

ieee_semantics! {
    Half = HalfS(16:5),
    Single = SingleS(32:8),
    Double = DoubleS(64:11),
    Quad = QuadS(128:15)
}

pub struct X87DoubleExtendedS;
pub type X87DoubleExtended = IeeeFloat<X87DoubleExtendedS>;
impl Semantics for X87DoubleExtendedS {
    const BITS: usize = 80;
    const PRECISION: usize = 64;
    const MAX_EXP: ExpInt = (1 << (15 - 1)) - 1;
}

float_common_impls!(IeeeFloat<S>);

impl<S: Semantics> PartialEq for IeeeFloat<S> {
    fn eq(&self, rhs: &Self) -> bool {
        self.partial_cmp(rhs) == Some(Ordering::Equal)
    }
}

#[allow(unused)]
impl<S: Semantics> PartialOrd for IeeeFloat<S> {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        panic!("NYI PartialOrd::partial_cmp");
    }
}

impl<S> Neg for IeeeFloat<S> {
    type Output = Self;
    fn neg(self) -> Self {
        panic!("NYI Neg::neg");
    }
}

/// Prints this value as a decimal string.
///
/// \param precision The maximum number of digits of
///   precision to output. If there are fewer digits available,
///   zero padding will not be used unless the value is
///   integral and small enough to be expressed in
///   precision digits. 0 means to use the natural
///   precision of the number.
/// \param width The maximum number of zeros to
///   consider inserting before falling back to scientific
///   notation. 0 means to always use scientific notation.
///
/// \param alternate Indicate whether to remove the trailing zero in
///   fraction part or not. Also setting this parameter to true forces
///   producing of output more similar to default printf behavior.
///   Specifically the lower e is used as exponent delimiter and exponent
///   always contains no less than two digits.
///
/// Number       precision    width      Result
/// ------       ---------    -----      ------
/// 1.01E+4              5        2       10100
/// 1.01E+4              4        2       1.01E+4
/// 1.01E+4              5        1       1.01E+4
/// 1.01E-2              5        2       0.0101
/// 1.01E-2              4        2       0.0101
/// 1.01E-2              4        1       1.01E-2
#[allow(unused)]
impl<S: Semantics> fmt::Display for IeeeFloat<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let frac_digits = f.precision().unwrap_or(0);
        let width = f.width().unwrap_or(3);
        let alternate = f.alternate();
        panic!("NYI Display::fmt");
    }
}

impl<S: Semantics> fmt::Debug for IeeeFloat<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

#[allow(unused)]
impl<S: Semantics> Float for IeeeFloat<S> {
    const BITS: usize = S::BITS;
    const PRECISION: usize = S::PRECISION;
    const MAX_EXP: ExpInt = S::MAX_EXP;
    const MIN_EXP: ExpInt = S::MIN_EXP;

    const ZERO: Self = IeeeFloat { marker: PhantomData };

    const INFINITY: Self = IeeeFloat { marker: PhantomData };

    // FIXME(eddyb) remove when qnan becomes const fn.
    const NAN: Self = IeeeFloat { marker: PhantomData };

    fn qnan(payload: Option<u128>) -> Self {
        panic!("NYI qnan")
    }

    fn snan(payload: Option<u128>) -> Self {
        panic!("NYI snan")
    }

    fn largest() -> Self {
        panic!("NYI largest")
    }

    const SMALLEST: Self = IeeeFloat { marker: PhantomData };

    fn smallest_normalized() -> Self {
        panic!("NYI smallest_normalized")
    }

    fn add_r(self, rhs: Self, round: Round) -> StatusAnd<Self> {
        panic!("NYI add_r")
    }

    fn mul_r(self, rhs: Self, round: Round) -> StatusAnd<Self> {
        panic!("NYI mul_r")
    }

    fn mul_add_r(self, multiplicand: Self, addend: Self, round: Round) -> StatusAnd<Self> {
        panic!("NYI mul_add_r")
    }

    fn div_r(self, rhs: Self, round: Round) -> StatusAnd<Self> {
        panic!("NYI div_r")
    }

    fn c_fmod(self, rhs: Self) -> StatusAnd<Self> {
        panic!("NYI c_fmod")
    }

    fn round_to_integral(self, round: Round) -> StatusAnd<Self> {
        panic!("NYI round_to_integral")
    }

    fn next_up(self) -> StatusAnd<Self> {
        panic!("NYI next_up")
    }

    fn from_bits(input: u128) -> Self {
        panic!("NYI from_bits")
    }

    fn from_u128_r(input: u128, round: Round) -> StatusAnd<Self> {
        panic!("NYI from_u128_r")
    }

    fn from_str_r(s: &str, round: Round) -> Result<StatusAnd<Self>, ParseError> {
        panic!("NYI from_str_r")
    }

    fn to_bits(self) -> u128 {
        panic!("NYI to_bits")
    }

    fn to_u128_r(self, width: usize, round: Round, is_exact: &mut bool) -> StatusAnd<u128> {
        panic!("NYI to_u128_r");
    }

    fn cmp_abs_normal(self, rhs: Self) -> Ordering {
        panic!("NYI cmp_abs_normal")
    }

    fn bitwise_eq(self, rhs: Self) -> bool {
        panic!("NYI bitwise_eq")
    }

    fn is_negative(self) -> bool {
        panic!("NYI is_negative")
    }

    fn is_denormal(self) -> bool {
        panic!("NYI is_denormal")
    }

    fn is_signaling(self) -> bool {
        panic!("NYI is_signaling")
    }

    fn category(self) -> Category {
        panic!("NYI category")
    }

    fn get_exact_inverse(self) -> Option<Self> {
        panic!("NYI get_exact_inverse")
    }

    fn ilogb(self) -> ExpInt {
        panic!("NYI ilogb")
    }

    fn scalbn_r(self, exp: ExpInt, round: Round) -> Self {
        panic!("NYI scalbn")
    }

    fn frexp_r(self, exp: &mut ExpInt, round: Round) -> Self {
        panic!("NYI frexp")
    }
}

#[allow(unused)]
impl<S: Semantics, T: Semantics> FloatConvert<IeeeFloat<T>> for IeeeFloat<S> {
    fn convert_r(self, round: Round, loses_info: &mut bool) -> StatusAnd<IeeeFloat<T>> {
        panic!("NYI convert_r");
    }
}
