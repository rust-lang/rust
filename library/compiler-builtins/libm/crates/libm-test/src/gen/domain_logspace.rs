//! A generator that produces logarithmically spaced values within domain bounds.

use std::ops::RangeInclusive;

use libm::support::{IntTy, MinInt};

use crate::domain::HasDomain;
use crate::op::OpITy;
use crate::{CheckCtx, MathOp, logspace};

/// Number of tests to run.
// FIXME(ntests): replace this with a more logical algorithm
const NTESTS: usize = {
    if cfg!(optimizations_enabled) {
        if crate::emulated()
            || !cfg!(target_pointer_width = "64")
            || cfg!(all(target_arch = "x86_64", target_vendor = "apple"))
        {
            // Tests are pretty slow on non-64-bit targets, x86 MacOS, and targets that run
            // in QEMU.
            100_000
        } else {
            5_000_000
        }
    } else {
        // Without optimizations just run a quick check
        800
    }
};

/// Create a range of logarithmically spaced inputs within a function's domain.
///
/// This allows us to get reasonably thorough coverage without wasting time on values that are
/// NaN or out of range. Random tests will still cover values that are excluded here.
pub fn get_test_cases<Op>(_ctx: &CheckCtx) -> impl Iterator<Item = (Op::FTy,)>
where
    Op: MathOp + HasDomain<Op::FTy>,
    IntTy<Op::FTy>: TryFrom<usize>,
    RangeInclusive<IntTy<Op::FTy>>: Iterator,
{
    let domain = Op::DOMAIN;
    let start = domain.range_start();
    let end = domain.range_end();
    let steps = OpITy::<Op>::try_from(NTESTS).unwrap_or(OpITy::<Op>::MAX);
    logspace(start, end, steps).map(|v| (v,))
}
