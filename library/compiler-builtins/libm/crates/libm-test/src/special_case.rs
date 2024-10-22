//! Configuration for skipping or changing the result for individual test cases (inputs) rather
//! than ignoring entire tests.

use crate::{CheckCtx, Float, Int, TestResult};

/// Type implementing [`IgnoreCase`].
pub struct SpecialCase;

/// Don't run further validation on this test case.
const SKIP: Option<TestResult> = Some(Ok(()));

/// Return this to skip checks on a test that currently fails but shouldn't. Looks
/// the same as skip, but we keep them separate to better indicate purpose.
const XFAIL: Option<TestResult> = Some(Ok(()));

/// Allow overriding the outputs of specific test cases.
///
/// There are some cases where we want to xfail specific cases or handle certain inputs
/// differently than the rest of calls to `validate`. This provides a hook to do that.
///
/// If `None` is returned, checks will proceed as usual. If `Some(result)` is returned, checks
/// are skipped and the provided result is returned instead.
///
/// This gets implemented once per input type, then the functions provide further filtering
/// based on function name and values.
///
/// `ulp` can also be set to adjust the ULP for that specific test, even if `None` is still
/// returned.
pub trait MaybeOverride<Input> {
    fn check_float<F: Float>(
        _input: Input,
        _actual: F,
        _expected: F,
        _ulp: &mut u32,
        _ctx: &CheckCtx,
    ) -> Option<TestResult> {
        None
    }

    fn check_int<I: Int>(
        _input: Input,
        _actual: I,
        _expected: I,
        _ctx: &CheckCtx,
    ) -> Option<TestResult> {
        None
    }
}

impl MaybeOverride<(f32,)> for SpecialCase {
    fn check_float<F: Float>(
        _input: (f32,),
        actual: F,
        expected: F,
        _ulp: &mut u32,
        ctx: &CheckCtx,
    ) -> Option<TestResult> {
        maybe_check_nan_bits(actual, expected, ctx)
    }
}

impl MaybeOverride<(f64,)> for SpecialCase {
    fn check_float<F: Float>(
        _input: (f64,),
        actual: F,
        expected: F,
        _ulp: &mut u32,
        ctx: &CheckCtx,
    ) -> Option<TestResult> {
        maybe_check_nan_bits(actual, expected, ctx)
    }
}

impl MaybeOverride<(f32, f32)> for SpecialCase {}
impl MaybeOverride<(f64, f64)> for SpecialCase {}
impl MaybeOverride<(f32, f32, f32)> for SpecialCase {}
impl MaybeOverride<(f64, f64, f64)> for SpecialCase {}
impl MaybeOverride<(i32, f32)> for SpecialCase {}
impl MaybeOverride<(i32, f64)> for SpecialCase {}
impl MaybeOverride<(f32, i32)> for SpecialCase {}
impl MaybeOverride<(f64, i32)> for SpecialCase {}

/// Check NaN bits if the function requires it
fn maybe_check_nan_bits<F: Float>(actual: F, expected: F, ctx: &CheckCtx) -> Option<TestResult> {
    if !(ctx.canonical_name == "abs" || ctx.canonical_name == "copysigh") {
        return None;
    }

    // abs and copysign require signaling NaNs to be propagated, so verify bit equality.
    if actual.to_bits() == expected.to_bits() {
        return SKIP;
    } else {
        Some(Err(anyhow::anyhow!("NaNs have different bitpatterns")))
    }
}
