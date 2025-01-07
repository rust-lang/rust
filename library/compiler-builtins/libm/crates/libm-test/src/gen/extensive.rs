use std::fmt;
use std::ops::RangeInclusive;

use libm::support::{Float, MinInt};

use crate::domain::HasDomain;
use crate::op::OpITy;
use crate::run_cfg::{int_range, iteration_count};
use crate::{CheckCtx, GeneratorKind, MathOp, linear_ints, logspace};

/// Generate a sequence of inputs that either cover the domain in completeness (for smaller float
/// types and single argument functions) or provide evenly spaced inputs across the domain with
/// approximately `u32::MAX` total iterations.
pub trait ExtensiveInput<Op> {
    fn get_cases(ctx: &CheckCtx) -> (impl Iterator<Item = Self> + Send, u64);
}

/// Construct an iterator from `logspace` and also calculate the total number of steps expected
/// for that iterator.
fn logspace_steps<Op>(
    start: Op::FTy,
    end: Op::FTy,
    max_steps: u64,
) -> (impl Iterator<Item = Op::FTy> + Clone, u64)
where
    Op: MathOp,
    OpITy<Op>: TryFrom<u64, Error: fmt::Debug>,
    u64: TryFrom<OpITy<Op>, Error: fmt::Debug>,
    RangeInclusive<OpITy<Op>>: Iterator,
{
    let max_steps = OpITy::<Op>::try_from(max_steps).unwrap_or(OpITy::<Op>::MAX);
    let (iter, steps) = logspace(start, end, max_steps);

    // `steps` will be <= the original `max_steps`, which is a `u64`.
    (iter, steps.try_into().unwrap())
}

/// Represents the iterator in either `Left` or `Right`.
enum EitherIter<A, B> {
    A(A),
    B(B),
}

impl<T, A: Iterator<Item = T>, B: Iterator<Item = T>> Iterator for EitherIter<A, B> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::A(iter) => iter.next(),
            Self::B(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::A(iter) => iter.size_hint(),
            Self::B(iter) => iter.size_hint(),
        }
    }
}

/// Gets the total number of possible values, returning `None` if that number doesn't fit in a
/// `u64`.
fn value_count<F: Float>() -> Option<u64>
where
    u64: TryFrom<F::Int>,
{
    u64::try_from(F::Int::MAX).ok().and_then(|max| max.checked_add(1))
}

/// Returns an iterator of every possible value of type `F`.
fn all_values<F: Float>() -> impl Iterator<Item = F>
where
    RangeInclusive<F::Int>: Iterator<Item = F::Int>,
{
    (F::Int::MIN..=F::Int::MAX).map(|bits| F::from_bits(bits))
}

macro_rules! impl_extensive_input {
    ($fty:ty) => {
        impl<Op> ExtensiveInput<Op> for ($fty,)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
            Op: HasDomain<Op::FTy>,
        {
            fn get_cases(ctx: &CheckCtx) -> (impl Iterator<Item = Self>, u64) {
                let max_steps0 = iteration_count(ctx, GeneratorKind::Extensive, 0);
                // `f16` and `f32` can have exhaustive tests.
                match value_count::<Op::FTy>() {
                    Some(steps0) if steps0 <= max_steps0 => {
                        let iter0 = all_values();
                        let iter0 = iter0.map(|v| (v,));
                        (EitherIter::A(iter0), steps0)
                    }
                    _ => {
                        let start = Op::DOMAIN.range_start();
                        let end = Op::DOMAIN.range_end();
                        let (iter0, steps0) = logspace_steps::<Op>(start, end, max_steps0);
                        let iter0 = iter0.map(|v| (v,));
                        (EitherIter::B(iter0), steps0)
                    }
                }
            }
        }

        impl<Op> ExtensiveInput<Op> for ($fty, $fty)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> (impl Iterator<Item = Self>, u64) {
                let max_steps0 = iteration_count(ctx, GeneratorKind::Extensive, 0);
                let max_steps1 = iteration_count(ctx, GeneratorKind::Extensive, 1);
                // `f16` can have exhaustive tests.
                match value_count::<Op::FTy>() {
                    Some(count) if count <= max_steps0 && count <= max_steps1 => {
                        let iter = all_values()
                            .flat_map(|first| all_values().map(move |second| (first, second)));
                        (EitherIter::A(iter), count.checked_mul(count).unwrap())
                    }
                    _ => {
                        let start = <$fty>::NEG_INFINITY;
                        let end = <$fty>::INFINITY;
                        let (iter0, steps0) = logspace_steps::<Op>(start, end, max_steps0);
                        let (iter1, steps1) = logspace_steps::<Op>(start, end, max_steps1);
                        let iter = iter0.flat_map(move |first| {
                            iter1.clone().map(move |second| (first, second))
                        });
                        let count = steps0.checked_mul(steps1).unwrap();
                        (EitherIter::B(iter), count)
                    }
                }
            }
        }

        impl<Op> ExtensiveInput<Op> for ($fty, $fty, $fty)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> (impl Iterator<Item = Self>, u64) {
                let max_steps0 = iteration_count(ctx, GeneratorKind::Extensive, 0);
                let max_steps1 = iteration_count(ctx, GeneratorKind::Extensive, 1);
                let max_steps2 = iteration_count(ctx, GeneratorKind::Extensive, 2);
                // `f16` can be exhaustive tested if `LIBM_EXTENSIVE_TESTS` is incresed.
                match value_count::<Op::FTy>() {
                    Some(count)
                        if count <= max_steps0 && count <= max_steps1 && count <= max_steps2 =>
                    {
                        let iter = all_values().flat_map(|first| {
                            all_values().flat_map(move |second| {
                                all_values().map(move |third| (first, second, third))
                            })
                        });
                        (EitherIter::A(iter), count.checked_pow(3).unwrap())
                    }
                    _ => {
                        let start = <$fty>::NEG_INFINITY;
                        let end = <$fty>::INFINITY;

                        let (iter0, steps0) = logspace_steps::<Op>(start, end, max_steps0);
                        let (iter1, steps1) = logspace_steps::<Op>(start, end, max_steps1);
                        let (iter2, steps2) = logspace_steps::<Op>(start, end, max_steps2);

                        let iter = iter0
                            .flat_map(move |first| iter1.clone().map(move |second| (first, second)))
                            .flat_map(move |(first, second)| {
                                iter2.clone().map(move |third| (first, second, third))
                            });
                        let count =
                            steps0.checked_mul(steps1).unwrap().checked_mul(steps2).unwrap();

                        (EitherIter::B(iter), count)
                    }
                }
            }
        }

        impl<Op> ExtensiveInput<Op> for (i32, $fty)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> (impl Iterator<Item = Self>, u64) {
                let range0 = int_range(ctx, GeneratorKind::Extensive, 0);
                let max_steps0 = iteration_count(ctx, GeneratorKind::Extensive, 0);
                let max_steps1 = iteration_count(ctx, GeneratorKind::Extensive, 1);
                match value_count::<Op::FTy>() {
                    Some(count1) if count1 <= max_steps1 => {
                        let (iter0, steps0) = linear_ints(range0, max_steps0);
                        let iter = iter0
                            .flat_map(move |first| all_values().map(move |second| (first, second)));
                        (EitherIter::A(iter), steps0.checked_mul(count1).unwrap())
                    }
                    _ => {
                        let start = <$fty>::NEG_INFINITY;
                        let end = <$fty>::INFINITY;

                        let (iter0, steps0) = linear_ints(range0, max_steps0);
                        let (iter1, steps1) = logspace_steps::<Op>(start, end, max_steps1);

                        let iter = iter0.flat_map(move |first| {
                            iter1.clone().map(move |second| (first, second))
                        });
                        let count = steps0.checked_mul(steps1).unwrap();

                        (EitherIter::B(iter), count)
                    }
                }
            }
        }

        impl<Op> ExtensiveInput<Op> for ($fty, i32)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> (impl Iterator<Item = Self>, u64) {
                let max_steps0 = iteration_count(ctx, GeneratorKind::Extensive, 0);
                let range1 = int_range(ctx, GeneratorKind::Extensive, 1);
                let max_steps1 = iteration_count(ctx, GeneratorKind::Extensive, 1);
                match value_count::<Op::FTy>() {
                    Some(count0) if count0 <= max_steps0 => {
                        let (iter1, steps1) = linear_ints(range1, max_steps1);
                        let iter = all_values().flat_map(move |first| {
                            iter1.clone().map(move |second| (first, second))
                        });
                        (EitherIter::A(iter), count0.checked_mul(steps1).unwrap())
                    }
                    _ => {
                        let start = <$fty>::NEG_INFINITY;
                        let end = <$fty>::INFINITY;

                        let (iter0, steps0) = logspace_steps::<Op>(start, end, max_steps0);
                        let (iter1, steps1) = linear_ints(range1, max_steps1);

                        let iter = iter0.flat_map(move |first| {
                            iter1.clone().map(move |second| (first, second))
                        });
                        let count = steps0.checked_mul(steps1).unwrap();

                        (EitherIter::B(iter), count)
                    }
                }
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

/// Create a test case iterator for extensive inputs. Also returns the total test case count.
pub fn get_test_cases<Op>(
    ctx: &CheckCtx,
) -> (impl Iterator<Item = Op::RustArgs> + Send + use<'_, Op>, u64)
where
    Op: MathOp,
    Op::RustArgs: ExtensiveInput<Op>,
{
    Op::RustArgs::get_cases(ctx)
}
