//! A generator that checks a handful of cases near infinities, zeros, asymptotes, and NaNs.

use libm::support::{CastInto, Float, Int, MinInt};

use crate::domain::get_domain;
use crate::generate::KnownSize;
use crate::op::OpITy;
use crate::run_cfg::{check_near_count, check_point_count};
use crate::{BaseName, CheckCtx, FloatExt, FloatTy, MathOp, test_log};

/// Generate a sequence of edge cases, e.g. numbers near zeroes and infiniteis.
pub trait EdgeCaseInput<Op> {
    fn get_cases(ctx: &CheckCtx) -> (impl Iterator<Item = Self> + Send, u64);
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
    let one = OpITy::<Op>::ONE;
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

    // Check around the maximum subnormal value
    let sub_max = Op::FTy::from_bits(Op::FTy::SIG_MASK);
    count_up(sub_max, near_points, values);
    count_down(sub_max, near_points, values);
    count_up(-sub_max, near_points, values);
    count_down(-sub_max, near_points, values);

    // Check a few values around the subnormal range
    for shift in (0..Op::FTy::SIG_BITS).step_by(Op::FTy::SIG_BITS as usize / 5) {
        let v = Op::FTy::from_bits(one << shift);
        count_up(v, 2, values);
        count_down(v, 2, values);
        count_up(-v, 2, values);
        count_down(-v, 2, values);
    }

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

/// Add `points` values starting at and including `x` and counting up. Uses the smallest possible
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

/// Add `points` values starting at and including `x` and counting down. Uses the smallest possible
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
    argnum: usize,
) -> (impl Iterator<Item = I> + Clone, u64)
where
    i32: CastInto<I>,
{
    let mut values = Vec::new();
    let near_points = check_near_count(ctx);

    // Check around max/min and zero
    int_count_around(I::MIN, near_points, &mut values);
    int_count_around(I::MAX, near_points, &mut values);
    int_count_around(I::ZERO, near_points, &mut values);
    int_count_around(I::ZERO, near_points, &mut values);

    if matches!(ctx.base_name, BaseName::Scalbn | BaseName::Ldexp) {
        assert_eq!(argnum, 1, "scalbn integer argument should be arg1");
        let (emax, emin, emin_sn) = match ctx.fn_ident.math_op().float_ty {
            FloatTy::F16 => {
                #[cfg(not(f16_enabled))]
                unreachable!();
                #[cfg(f16_enabled)]
                (f16::EXP_MAX, f16::EXP_MIN, f16::EXP_MIN_SUBNORM)
            }
            FloatTy::F32 => (f32::EXP_MAX, f32::EXP_MIN, f32::EXP_MIN_SUBNORM),
            FloatTy::F64 => (f64::EXP_MAX, f64::EXP_MIN, f64::EXP_MIN_SUBNORM),
            FloatTy::F128 => {
                #[cfg(not(f128_enabled))]
                unreachable!();
                #[cfg(f128_enabled)]
                (f128::EXP_MAX, f128::EXP_MIN, f128::EXP_MIN_SUBNORM)
            }
        };

        // `scalbn`/`ldexp` have their trickiest behavior around exponent limits
        int_count_around(emax.cast(), near_points, &mut values);
        int_count_around(emin.cast(), near_points, &mut values);
        int_count_around(emin_sn.cast(), near_points, &mut values);
        int_count_around((-emin_sn).cast(), near_points, &mut values);

        // Also check values that cause the maximum possible difference in exponents
        int_count_around((emax - emin).cast(), near_points, &mut values);
        int_count_around((emin - emax).cast(), near_points, &mut values);
        int_count_around((emax - emin_sn).cast(), near_points, &mut values);
        int_count_around((emin_sn - emax).cast(), near_points, &mut values);
    }

    values.sort();
    values.dedup();
    let count = values.len().try_into().unwrap();

    test_log(&format!(
        "{gen_kind:?} {basis:?} {fn_ident} arg {arg}/{args}: {count} edge cases",
        gen_kind = ctx.gen_kind,
        basis = ctx.basis,
        fn_ident = ctx.fn_ident,
        arg = argnum + 1,
        args = ctx.input_count(),
    ));

    (values.into_iter(), count)
}

/// Add `points` values both up and down, starting at and including `x`.
fn int_count_around<I: Int>(x: I, points: u64, values: &mut Vec<I>) {
    let mut current = x;
    for _ in 0..points {
        values.push(current);
        current = match current.checked_add(I::ONE) {
            Some(v) => v,
            None => break,
        };
    }

    current = x;
    for _ in 0..points {
        values.push(current);
        current = match current.checked_sub(I::ONE) {
            Some(v) => v,
            None => break,
        };
    }
}

macro_rules! impl_edge_case_input {
    ($fty:ty) => {
        impl<Op> EdgeCaseInput<Op> for ($fty,)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> (impl Iterator<Item = Self>, u64) {
                let (iter0, steps0) = float_edge_cases::<Op>(ctx, 0);
                let iter0 = iter0.map(|v| (v,));
                (iter0, steps0)
            }
        }

        impl<Op> EdgeCaseInput<Op> for ($fty, $fty)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> (impl Iterator<Item = Self>, u64) {
                let (iter0, steps0) = float_edge_cases::<Op>(ctx, 0);
                let (iter1, steps1) = float_edge_cases::<Op>(ctx, 1);
                let iter =
                    iter0.flat_map(move |first| iter1.clone().map(move |second| (first, second)));
                let count = steps0.checked_mul(steps1).unwrap();
                (iter, count)
            }
        }

        impl<Op> EdgeCaseInput<Op> for ($fty, $fty, $fty)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> (impl Iterator<Item = Self>, u64) {
                let (iter0, steps0) = float_edge_cases::<Op>(ctx, 0);
                let (iter1, steps1) = float_edge_cases::<Op>(ctx, 1);
                let (iter2, steps2) = float_edge_cases::<Op>(ctx, 2);

                let iter = iter0
                    .flat_map(move |first| iter1.clone().map(move |second| (first, second)))
                    .flat_map(move |(first, second)| {
                        iter2.clone().map(move |third| (first, second, third))
                    });
                let count = steps0.checked_mul(steps1).unwrap().checked_mul(steps2).unwrap();

                (iter, count)
            }
        }

        impl<Op> EdgeCaseInput<Op> for (i32, $fty)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> (impl Iterator<Item = Self>, u64) {
                let (iter0, steps0) = int_edge_cases(ctx, 0);
                let (iter1, steps1) = float_edge_cases::<Op>(ctx, 1);

                let iter =
                    iter0.flat_map(move |first| iter1.clone().map(move |second| (first, second)));
                let count = steps0.checked_mul(steps1).unwrap();

                (iter, count)
            }
        }

        impl<Op> EdgeCaseInput<Op> for ($fty, i32)
        where
            Op: MathOp<RustArgs = Self, FTy = $fty>,
        {
            fn get_cases(ctx: &CheckCtx) -> (impl Iterator<Item = Self>, u64) {
                let (iter0, steps0) = float_edge_cases::<Op>(ctx, 0);
                let (iter1, steps1) = int_edge_cases(ctx, 1);

                let iter =
                    iter0.flat_map(move |first| iter1.clone().map(move |second| (first, second)));
                let count = steps0.checked_mul(steps1).unwrap();

                (iter, count)
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
) -> (impl Iterator<Item = Op::RustArgs> + Send + use<'_, Op>, u64)
where
    Op: MathOp,
    Op::RustArgs: EdgeCaseInput<Op>,
{
    let (iter, count) = Op::RustArgs::get_cases(ctx);

    // Wrap in `KnownSize` so we get an assertion if the cuunt is wrong.
    (KnownSize::new(iter, count), count)
}
