//! A generator that checks a handful of cases near infinities, zeros, asymptotes, and NaNs.

use libm::support::{Float, Int};

use crate::domain::get_domain;
use crate::gen::KnownSize;
use crate::run_cfg::{check_near_count, check_point_count};
use crate::{CheckCtx, FloatExt, MathOp, test_log};

/// Generate a sequence of edge cases, e.g. numbers near zeroes and infiniteis.
pub trait EdgeCaseInput<Op> {
    fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> + Send;
}

/// Create a list of values around interesting points (infinities, zeroes, NaNs).
fn float_edge_cases<Op>(
    ctx: &CheckCtx,
    argnum: usize,
) -> (impl Iterator<Item = Op::FTy> + Clone, u64)
where
    Op: MathOp,
{
    let mut ret = Vec::new();
    let values = &mut ret;
    let domain = get_domain::<_, i8>(ctx.fn_ident, argnum).unwrap_float();
    let domain_start = domain.range_start();
    let domain_end = domain.range_end();

    let check_points = check_point_count(ctx);
    let near_points = check_near_count(ctx);

    // Check near some notable constants
    count_up(Op::FTy::ONE, near_points, values);
    count_up(Op::FTy::ZERO, near_points, values);
    count_up(Op::FTy::NEG_ONE, near_points, values);
    count_down(Op::FTy::ONE, near_points, values);
    count_down(Op::FTy::ZERO, near_points, values);
    count_down(Op::FTy::NEG_ONE, near_points, values);
    values.push(Op::FTy::NEG_ZERO);

    // Check values near the extremes
    count_up(Op::FTy::NEG_INFINITY, near_points, values);
    count_down(Op::FTy::INFINITY, near_points, values);
    count_down(domain_end, near_points, values);
    count_up(domain_start, near_points, values);
    count_down(domain_start, near_points, values);
    count_up(domain_end, near_points, values);
    count_down(domain_end, near_points, values);

    // Check some special values that aren't included in the above ranges
    values.push(Op::FTy::NAN);
    values.extend(Op::FTy::consts().iter());

    // Check around asymptotes
    if let Some(f) = domain.check_points {
        let iter = f();
        for x in iter.take(check_points) {
            count_up(x, near_points, values);
            count_down(x, near_points, values);
        }
    }

    // Some results may overlap so deduplicate the vector to save test cycles.
    values.sort_by_key(|x| x.to_bits());
    values.dedup_by_key(|x| x.to_bits());

    let count = ret.len().try_into().unwrap();

    test_log(&format!(
        "{gen_kind:?} {basis:?} {fn_ident} arg {arg}/{args}: {count} edge cases",
        gen_kind = ctx.gen_kind,
        basis = ctx.basis,
        fn_ident = ctx.fn_ident,
        arg = argnum + 1,
        args = ctx.input_count(),
    ));

    (ret.into_iter(), count)
}

/// Add `AROUND` values starting at and including `x` and counting up. Uses the smallest possible
/// increments (1 ULP).
fn count_up<F: Float>(mut x: F, points: u64, values: &mut Vec<F>) {
    assert!(!x.is_nan());

    let mut count = 0;
    while x < F::INFINITY && count < points {
        values.push(x);
        x = x.next_up();
        count += 1;
    }
}

/// Add `AROUND` values starting at and including `x` and counting down. Uses the smallest possible
/// increments (1 ULP).
fn count_down<F: Float>(mut x: F, points: u64, values: &mut Vec<F>) {
    assert!(!x.is_nan());

    let mut count = 0;
    while x > F::NEG_INFINITY && count < points {
        values.push(x);
        x = x.next_down();
        count += 1;
    }
}

/// Create a list of values around interesting integer points (min, zero, max).
pub fn int_edge_cases<I: Int>(
    ctx: &CheckCtx,
    _argnum: usize,
) -> (impl Iterator<Item = I> + Clone, u64) {
    let mut values = Vec::new();
    let near_points = check_near_count(ctx);

    for up_from in [I::MIN, I::ZERO] {
        let mut x = up_from;
        for _ in 0..near_points {
            values.push(x);
            x += I::ONE;
        }
    }

    for down_from in [I::ZERO, I::MAX] {
        let mut x = down_from;
        for _ in 0..near_points {
            values.push(x);
            x -= I::ONE;
        }
    }

    values.sort();
    values.dedup();
    let len = values.len().try_into().unwrap();
    (values.into_iter(), len)
}

macro_rules! impl_edge_case_input {
    ($fty:ty) => {
        impl<Op> EdgeCaseInput<Op> for ($fty,)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let (iter0, steps0) = float_edge_cases::<Op>(ctx, 0);
                let iter0 = iter0.map(|v| (v,));
                KnownSize::new(iter0, steps0)
            }
        }

        impl<Op> EdgeCaseInput<Op> for ($fty, $fty)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let (iter0, steps0) = float_edge_cases::<Op>(ctx, 0);
                let (iter1, steps1) = float_edge_cases::<Op>(ctx, 1);
                let iter =
                    iter0.flat_map(move |first| iter1.clone().map(move |second| (first, second)));
                let count = steps0.checked_mul(steps1).unwrap();
                KnownSize::new(iter, count)
            }
        }

        impl<Op> EdgeCaseInput<Op> for ($fty, $fty, $fty)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let (iter0, steps0) = float_edge_cases::<Op>(ctx, 0);
                let (iter1, steps1) = float_edge_cases::<Op>(ctx, 1);
                let (iter2, steps2) = float_edge_cases::<Op>(ctx, 2);

                let iter = iter0
                    .flat_map(move |first| iter1.clone().map(move |second| (first, second)))
                    .flat_map(move |(first, second)| {
                        iter2.clone().map(move |third| (first, second, third))
                    });
                let count = steps0.checked_mul(steps1).unwrap().checked_mul(steps2).unwrap();

                KnownSize::new(iter, count)
            }
        }

        impl<Op> EdgeCaseInput<Op> for (i32, $fty)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let (iter0, steps0) = int_edge_cases(ctx, 0);
                let (iter1, steps1) = float_edge_cases::<Op>(ctx, 1);

                let iter =
                    iter0.flat_map(move |first| iter1.clone().map(move |second| (first, second)));
                let count = steps0.checked_mul(steps1).unwrap();

                KnownSize::new(iter, count)
            }
        }

        impl<Op> EdgeCaseInput<Op> for ($fty, i32)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let (iter0, steps0) = float_edge_cases::<Op>(ctx, 0);
                let (iter1, steps1) = int_edge_cases(ctx, 1);

                let iter =
                    iter0.flat_map(move |first| iter1.clone().map(move |second| (first, second)));
                let count = steps0.checked_mul(steps1).unwrap();

                KnownSize::new(iter, count)
            }
        }
    };
}

#[cfg(f16_enabled)]
impl_edge_case_input!(f16);
impl_edge_case_input!(f32);
impl_edge_case_input!(f64);
#[cfg(f128_enabled)]
impl_edge_case_input!(f128);

pub fn get_test_cases<Op>(
    ctx: &CheckCtx,
) -> impl ExactSizeIterator<Item = Op::RustArgs> + use<'_, Op>
where
    Op: MathOp,
    Op::RustArgs: EdgeCaseInput<Op>,
{
    Op::RustArgs::get_cases(ctx)
}
