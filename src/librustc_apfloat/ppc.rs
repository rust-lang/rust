// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use {ieee, Category, ExpInt, Float, Round, ParseError, StatusAnd};

use std::cmp::Ordering;
use std::fmt;
use std::ops::Neg;

#[must_use]
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct DoubleFloat<F>(F, F);
pub type DoubleDouble = DoubleFloat<ieee::Double>;

// These are legacy semantics for the Fallback, inaccrurate implementation of
// IBM double-double, if the accurate DoubleDouble doesn't handle the
// operation. It's equivalent to having an IEEE number with consecutive 106
// bits of mantissa and 11 bits of exponent.
//
// It's not equivalent to IBM double-double. For example, a legit IBM
// double-double, 1 + epsilon:
//
//   1 + epsilon = 1 + (1 >> 1076)
//
// is not representable by a consecutive 106 bits of mantissa.
//
// Currently, these semantics are used in the following way:
//
//   DoubleDouble -> (Double, Double) ->
//   DoubleDouble's Fallback -> IEEE operations
//
// FIXME: Implement all operations in DoubleDouble, and delete these
// semantics.
// FIXME(eddyb) This shouldn't need to be `pub`, it's only used in bounds.
pub struct FallbackS<F>(F);
type Fallback<F> = ieee::IeeeFloat<FallbackS<F>>;
impl<F: Float> ieee::Semantics for FallbackS<F> {
    // Forbid any conversion to/from bits.
    const BITS: usize = 0;
    const PRECISION: usize = F::PRECISION * 2;
    const MAX_EXP: ExpInt = F::MAX_EXP as ExpInt;
    const MIN_EXP: ExpInt = F::MIN_EXP as ExpInt + F::PRECISION as ExpInt;
}

float_common_impls!(DoubleFloat<F>);

impl<F: Float> Neg for DoubleFloat<F> {
    type Output = Self;
    fn neg(self) -> Self {
        panic!("NYI Neg::neg");
    }
}

#[allow(unused)]
impl<F: Float> fmt::Display for DoubleFloat<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        panic!("NYI Display::fmt");
    }
}

#[allow(unused)]
impl<F: Float> Float for DoubleFloat<F> {
    const BITS: usize = F::BITS * 2;
    const PRECISION: usize = Fallback::<F>::PRECISION;
    const MAX_EXP: ExpInt = Fallback::<F>::MAX_EXP;
    const MIN_EXP: ExpInt = Fallback::<F>::MIN_EXP;

    const ZERO: Self = DoubleFloat(F::ZERO, F::ZERO);

    const INFINITY: Self = DoubleFloat(F::INFINITY, F::ZERO);

    // FIXME(eddyb) remove when qnan becomes const fn.
    const NAN: Self = DoubleFloat(F::NAN, F::ZERO);

    fn qnan(payload: Option<u128>) -> Self {
        panic!("NYI qnan")
    }

    fn snan(payload: Option<u128>) -> Self {
        panic!("NYI snan")
    }

    fn largest() -> Self {
        panic!("NYI largest")
    }

    const SMALLEST: Self = DoubleFloat(F::SMALLEST, F::ZERO);

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
