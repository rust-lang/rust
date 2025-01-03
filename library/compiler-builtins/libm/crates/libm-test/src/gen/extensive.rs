use std::fmt;
use std::ops::RangeInclusive;

use libm::support::MinInt;

use crate::domain::HasDomain;
use crate::gen::KnownSize;
use crate::op::OpITy;
use crate::run_cfg::{int_range, iteration_count};
use crate::{CheckCtx, GeneratorKind, MathOp, logspace};

/// Generate a sequence of inputs that either cover the domain in completeness (for smaller float
/// types and single argument functions) or provide evenly spaced inputs across the domain with
/// approximately `u32::MAX` total iterations.
pub trait ExtensiveInput<Op> {
    fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> + Send;
}

/// Construct an iterator from `logspace` and also calculate the total number of steps expected
/// for that iterator.
fn logspace_steps<Op>(
    start: Op::FTy,
    end: Op::FTy,
    ctx: &CheckCtx,
    argnum: usize,
) -> (impl Iterator<Item = Op::FTy> + Clone, u64)
where
    Op: MathOp,
    OpITy<Op>: TryFrom<u64, Error: fmt::Debug>,
    RangeInclusive<OpITy<Op>>: Iterator,
{
    let max_steps = iteration_count(ctx, GeneratorKind::Extensive, argnum);
    let max_steps = OpITy::<Op>::try_from(max_steps).unwrap_or(OpITy::<Op>::MAX);
    let iter = logspace(start, end, max_steps);

    // `logspace` can't implement `ExactSizeIterator` because of the range, but its size hint
    // should be accurate (assuming <= usize::MAX iterations).
    let size_hint = iter.size_hint();
    assert_eq!(size_hint.0, size_hint.1.unwrap());

    (iter, size_hint.0.try_into().unwrap())
}

macro_rules! impl_extensive_input {
    ($fty:ty) => {
        impl<Op> ExtensiveInput<Op> for ($fty,)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
            Op: HasDomain<Op::FTy>,
        {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let start = Op::DOMAIN.range_start();
                let end = Op::DOMAIN.range_end();
                let (iter0, steps0) = logspace_steps::<Op>(start, end, ctx, 0);
                let iter0 = iter0.map(|v| (v,));
                KnownSize::new(iter0, steps0)
            }
        }

        impl<Op> ExtensiveInput<Op> for ($fty, $fty)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let start = <$fty>::NEG_INFINITY;
                let end = <$fty>::INFINITY;
                let (iter0, steps0) = logspace_steps::<Op>(start, end, ctx, 0);
                let (iter1, steps1) = logspace_steps::<Op>(start, end, ctx, 1);
                let iter =
                    iter0.flat_map(move |first| iter1.clone().map(move |second| (first, second)));
                let count = steps0.checked_mul(steps1).unwrap();
                KnownSize::new(iter, count)
            }
        }

        impl<Op> ExtensiveInput<Op> for ($fty, $fty, $fty)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let start = <$fty>::NEG_INFINITY;
                let end = <$fty>::INFINITY;

                let (iter0, steps0) = logspace_steps::<Op>(start, end, ctx, 0);
                let (iter1, steps1) = logspace_steps::<Op>(start, end, ctx, 1);
                let (iter2, steps2) = logspace_steps::<Op>(start, end, ctx, 2);

                let iter = iter0
                    .flat_map(move |first| iter1.clone().map(move |second| (first, second)))
                    .flat_map(move |(first, second)| {
                        iter2.clone().map(move |third| (first, second, third))
                    });
                let count = steps0.checked_mul(steps1).unwrap().checked_mul(steps2).unwrap();

                KnownSize::new(iter, count)
            }
        }

        impl<Op> ExtensiveInput<Op> for (i32, $fty)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let start = <$fty>::NEG_INFINITY;
                let end = <$fty>::INFINITY;

                let iter0 = int_range(ctx, GeneratorKind::Extensive, 0);
                let steps0 = iteration_count(ctx, GeneratorKind::Extensive, 0);
                let (iter1, steps1) = logspace_steps::<Op>(start, end, ctx, 1);

                let iter =
                    iter0.flat_map(move |first| iter1.clone().map(move |second| (first, second)));
                let count = steps0.checked_mul(steps1).unwrap();

                KnownSize::new(iter, count)
            }
        }

        impl<Op> ExtensiveInput<Op> for ($fty, i32)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let start = <$fty>::NEG_INFINITY;
                let end = <$fty>::INFINITY;

                let (iter0, steps0) = logspace_steps::<Op>(start, end, ctx, 0);
                let iter1 = int_range(ctx, GeneratorKind::Extensive, 0);
                let steps1 = iteration_count(ctx, GeneratorKind::Extensive, 0);

                let iter =
                    iter0.flat_map(move |first| iter1.clone().map(move |second| (first, second)));
                let count = steps0.checked_mul(steps1).unwrap();

                KnownSize::new(iter, count)
            }
        }
    };
}

#[cfg(f16_enabled)]
impl_extensive_input!(f16);
impl_extensive_input!(f32);
impl_extensive_input!(f64);
#[cfg(f128_enabled)]
impl_extensive_input!(f128);

/// Create a test case iterator for extensive inputs.
pub fn get_test_cases<Op>(
    ctx: &CheckCtx,
) -> impl ExactSizeIterator<Item = Op::RustArgs> + Send + use<'_, Op>
where
    Op: MathOp,
    Op::RustArgs: ExtensiveInput<Op>,
{
    Op::RustArgs::get_cases(ctx)
}
