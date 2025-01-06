//! A generator that produces logarithmically spaced values within domain bounds.

use std::ops::RangeInclusive;

use libm::support::{IntTy, MinInt};

use crate::domain::HasDomain;
use crate::op::OpITy;
use crate::run_cfg::{GeneratorKind, iteration_count};
use crate::{CheckCtx, MathOp, logspace};

/// Create a range of logarithmically spaced inputs within a function's domain.
///
/// This allows us to get reasonably thorough coverage without wasting time on values that are
/// NaN or out of range. Random tests will still cover values that are excluded here.
pub fn get_test_cases<Op>(ctx: &CheckCtx) -> impl Iterator<Item = (Op::FTy,)>
where
    Op: MathOp + HasDomain<Op::FTy>,
    IntTy<Op::FTy>: TryFrom<u64>,
    RangeInclusive<IntTy<Op::FTy>>: Iterator,
{
    let domain = Op::DOMAIN;
    let ntests = iteration_count(ctx, GeneratorKind::Domain, 0);

    // We generate logspaced inputs within a specific range, excluding values that are out of
    // range in order to make iterations useful (random tests still cover the full range).
    let start = domain.range_start();
    let end = domain.range_end();
    let steps = OpITy::<Op>::try_from(ntests).unwrap_or(OpITy::<Op>::MAX);
    logspace(start, end, steps).map(|v| (v,))
}
