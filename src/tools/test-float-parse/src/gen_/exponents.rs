use std::fmt::Write;
use std::ops::RangeInclusive;

use crate::traits::BoxGenIter;
use crate::{Float, Generator};

const SMALL_COEFF_MAX: i32 = 10_000;
const SMALL_EXP_MAX: i32 = 300;

const SMALL_COEFF_RANGE: RangeInclusive<i32> = (-SMALL_COEFF_MAX)..=SMALL_COEFF_MAX;
const SMALL_EXP_RANGE: RangeInclusive<i32> = (-SMALL_EXP_MAX)..=SMALL_EXP_MAX;

const LARGE_COEFF_RANGE: RangeInclusive<u32> = 0..=100_000;
const LARGE_EXP_RANGE: RangeInclusive<u32> = 300..=350;

/// Check exponential values around zero.
pub struct SmallExponents<F: Float> {
    iter: BoxGenIter<Self, F>,
}

impl<F: Float> Generator<F> for SmallExponents<F> {
    const NAME: &'static str = "small exponents";
    const SHORT_NAME: &'static str = "small exp";

    /// `(coefficient, exponent)`
    type WriteCtx = (i32, i32);

    fn total_tests() -> u64 {
        ((1 + SMALL_COEFF_RANGE.end() - SMALL_COEFF_RANGE.start())
            * (1 + SMALL_EXP_RANGE.end() - SMALL_EXP_RANGE.start()))
        .try_into()
        .unwrap()
    }

    fn new() -> Self {
        let iter = SMALL_EXP_RANGE.flat_map(|exp| SMALL_COEFF_RANGE.map(move |coeff| (coeff, exp)));

        Self { iter: Box::new(iter) }
    }

    fn write_string(s: &mut String, ctx: Self::WriteCtx) {
        let (coeff, exp) = ctx;
        write!(s, "{coeff}e{exp}").unwrap();
    }
}

impl<F: Float> Iterator for SmallExponents<F> {
    type Item = (i32, i32);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

/// Check exponential values further from zero.
pub struct LargeExponents<F: Float> {
    iter: BoxGenIter<Self, F>,
}

impl<F: Float> Generator<F> for LargeExponents<F> {
    const NAME: &'static str = "large positive exponents";
    const SHORT_NAME: &'static str = "large exp";

    /// `(coefficient, exponent, is_positive)`
    type WriteCtx = (u32, u32, bool);

    fn total_tests() -> u64 {
        ((1 + LARGE_EXP_RANGE.end() - LARGE_EXP_RANGE.start())
            * (1 + LARGE_COEFF_RANGE.end() - LARGE_COEFF_RANGE.start())
            * 2)
        .into()
    }

    fn new() -> Self {
        let iter = LARGE_EXP_RANGE
            .flat_map(|exp| LARGE_COEFF_RANGE.map(move |coeff| (coeff, exp)))
            .flat_map(|(coeff, exp)| [(coeff, exp, false), (coeff, exp, true)]);

        Self { iter: Box::new(iter) }
    }

    fn write_string(s: &mut String, ctx: Self::WriteCtx) {
        let (coeff, exp, is_positive) = ctx;
        let sign = if is_positive { "" } else { "-" };
        write!(s, "{sign}{coeff}e{exp}").unwrap();
    }
}

impl<F: Float> Iterator for LargeExponents<F> {
    type Item = (u32, u32, bool);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
