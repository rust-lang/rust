use std::fmt::Write;
use std::ops::RangeInclusive;

use crate::{Float, Generator, Int};

/// Test every possible bit pattern. This is infeasible to run on any float types larger than
/// `f32` (which takes about an hour).
pub struct Exhaustive<F: Float> {
    iter: RangeInclusive<F::Int>,
}

impl<F: Float> Generator<F> for Exhaustive<F>
where
    RangeInclusive<F::Int>: Iterator<Item = F::Int>,
{
    const SHORT_NAME: &'static str = "exhaustive";

    type WriteCtx = F;

    fn total_tests() -> u64 {
        1u64.checked_shl(F::Int::BITS).expect("More than u64::MAX tests")
    }

    fn new() -> Self {
        Self { iter: F::Int::ZERO..=F::Int::MAX }
    }

    fn write_string(s: &mut String, ctx: Self::WriteCtx) {
        write!(s, "{ctx:e}").unwrap();
    }
}

impl<F: Float> Iterator for Exhaustive<F>
where
    RangeInclusive<F::Int>: Iterator<Item = F::Int>,
{
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        Some(F::from_bits(self.iter.next()?))
    }
}
