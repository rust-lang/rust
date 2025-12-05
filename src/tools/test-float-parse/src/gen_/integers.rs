use std::fmt::Write;
use std::ops::{Range, RangeInclusive};

use crate::traits::BoxGenIter;
use crate::{Float, Generator};

const SMALL_MAX_POW2: u32 = 19;

/// All values up to the max power of two
const SMALL_VALUES: RangeInclusive<i32> = {
    let max = 1i32 << SMALL_MAX_POW2;
    (-max)..=max
};

/// Large values only get tested around powers of two
const LARGE_POWERS: Range<u32> = SMALL_MAX_POW2..128;

/// We perturbe each large value around these ranges
const LARGE_PERTURBATIONS: RangeInclusive<i128> = -256..=256;

/// Test all integers up to `2 ^ MAX_POW2`
pub struct SmallInt {
    iter: RangeInclusive<i32>,
}

impl<F: Float> Generator<F> for SmallInt {
    const NAME: &'static str = "small integer values";
    const SHORT_NAME: &'static str = "int small";

    type WriteCtx = i32;

    fn total_tests() -> u64 {
        (SMALL_VALUES.end() + 1 - SMALL_VALUES.start()).try_into().unwrap()
    }

    fn new() -> Self {
        Self { iter: SMALL_VALUES }
    }

    fn write_string(s: &mut String, ctx: Self::WriteCtx) {
        write!(s, "{ctx}").unwrap();
    }
}

impl Iterator for SmallInt {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

/// Test much bigger integers than [`SmallInt`].
pub struct LargeInt<F: Float> {
    iter: BoxGenIter<Self, F>,
}

impl<F: Float> LargeInt<F> {
    const EDGE_CASES: [i128; 7] = [
        i32::MIN as i128,
        i32::MAX as i128,
        i64::MIN as i128,
        i64::MAX as i128,
        u64::MAX as i128,
        i128::MIN,
        i128::MAX,
    ];
}

impl<F: Float> Generator<F> for LargeInt<F> {
    const NAME: &'static str = "large integer values";
    const SHORT_NAME: &'static str = "int large";

    type WriteCtx = i128;

    fn total_tests() -> u64 {
        u64::try_from(
            (i128::from(LARGE_POWERS.end - LARGE_POWERS.start)
                + i128::try_from(Self::EDGE_CASES.len()).unwrap())
                * (LARGE_PERTURBATIONS.end() + 1 - LARGE_PERTURBATIONS.start()),
        )
        .unwrap()
    }

    fn new() -> Self {
        let iter = LARGE_POWERS
            .map(|pow| 1i128 << pow)
            .chain(Self::EDGE_CASES)
            .flat_map(|base| LARGE_PERTURBATIONS.map(move |perturb| base.saturating_add(perturb)));

        Self { iter: Box::new(iter) }
    }

    fn write_string(s: &mut String, ctx: Self::WriteCtx) {
        write!(s, "{ctx}").unwrap();
    }
}
impl<F: Float> Iterator for LargeInt<F> {
    type Item = i128;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
