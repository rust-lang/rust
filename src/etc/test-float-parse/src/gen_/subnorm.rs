use std::fmt::Write;
use std::ops::RangeInclusive;

use crate::{Float, Generator, Int};

/// Spot check some edge cases for subnormals.
pub struct SubnormEdgeCases<F: Float> {
    cases: [F::Int; 6],
    index: usize,
}

impl<F: Float> SubnormEdgeCases<F> {
    /// Shorthand
    const I1: F::Int = F::Int::ONE;

    fn edge_cases() -> [F::Int; 6] {
        // Comments use an 8-bit mantissa as a demo
        [
            // 0b00000001
            Self::I1,
            // 0b10000000
            Self::I1 << (F::MAN_BITS - 1),
            // 0b00001000
            Self::I1 << ((F::MAN_BITS / 2) - 1),
            // 0b00001111
            Self::I1 << ((F::MAN_BITS / 2) - 1),
            // 0b00001111
            Self::I1 << ((F::MAN_BITS / 2) - 1),
            // 0b11111111
            F::MAN_MASK,
        ]
    }
}

impl<F: Float> Generator<F> for SubnormEdgeCases<F> {
    const NAME: &'static str = "subnormal edge cases";
    const SHORT_NAME: &'static str = "subnorm edge";

    type WriteCtx = F;

    fn new() -> Self {
        Self { cases: Self::edge_cases(), index: 0 }
    }

    fn total_tests() -> u64 {
        Self::edge_cases().len().try_into().unwrap()
    }

    fn write_string(s: &mut String, ctx: Self::WriteCtx) {
        write!(s, "{ctx:e}").unwrap();
    }
}

impl<F: Float> Iterator for SubnormEdgeCases<F> {
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.cases.get(self.index)?;
        self.index += 1;

        Some(F::from_bits(*i))
    }
}

/// Test all subnormals up to `1 << 22`.
pub struct SubnormComplete<F: Float> {
    iter: RangeInclusive<F::Int>,
}

impl<F: Float> Generator<F> for SubnormComplete<F>
where
    RangeInclusive<F::Int>: Iterator<Item = F::Int>,
{
    const NAME: &'static str = "subnormal";
    const SHORT_NAME: &'static str = "subnorm ";

    type WriteCtx = F;

    fn total_tests() -> u64 {
        let iter = Self::new().iter;
        (F::Int::ONE + *iter.end() - *iter.start()).try_into().unwrap()
    }

    fn new() -> Self {
        let upper_lim = if F::MAN_BITS >= 22 {
            F::Int::ONE << 22
        } else {
            (F::Int::ONE << F::MAN_BITS) - F::Int::ONE
        };

        Self { iter: F::Int::ZERO..=upper_lim }
    }

    fn write_string(s: &mut String, ctx: Self::WriteCtx) {
        write!(s, "{ctx:e}").unwrap();
    }
}

impl<F: Float> Iterator for SubnormComplete<F>
where
    RangeInclusive<F::Int>: Iterator<Item = F::Int>,
{
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        Some(F::from_bits(self.iter.next()?))
    }
}
