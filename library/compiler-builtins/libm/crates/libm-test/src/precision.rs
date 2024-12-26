//! Configuration for skipping or changing the result for individual test cases (inputs) rather
//! than ignoring entire tests.

use core::f32;

use CheckBasis::{Mpfr, Musl};
use Identifier as Id;

use crate::{BaseName, CheckBasis, CheckCtx, Float, Identifier, Int, TestResult};

/// Type implementing [`IgnoreCase`].
pub struct SpecialCase;

/// Default ULP allowed to differ from musl (note that musl itself may not be accurate).
const MUSL_DEFAULT_ULP: u32 = 2;

/// Default ULP allowed to differ from multiprecision (i.e. infinite) results.
const MP_DEFAULT_ULP: u32 = 1;

/// ULP allowed to differ from the results returned by a test basis.
///
/// Note that these results were obtained using 400M rounds of random inputs, which
/// is not a value used by default.
pub fn default_ulp(ctx: &CheckCtx) -> u32 {
    match (&ctx.basis, ctx.fn_ident) {
        // Overrides that apply to either basis
        (_, Id::J0 | Id::J0f | Id::J1 | Id::J1f) => {
            // Results seem very target-dependent
            if cfg!(target_arch = "x86_64") { 4000 } else { 800_000 }
        }
        (_, Id::Jn | Id::Jnf) => 1000,

        // Overrides for musl
        #[cfg(x86_no_sse)]
        (Musl, Id::Asinh | Id::Asinhf) => 6,
        #[cfg(not(target_pointer_width = "64"))]
        (Musl, Id::Exp10 | Id::Exp10f) => 4,
        (Musl, Id::Lgamma | Id::LgammaR | Id::Lgammaf | Id::LgammafR) => 400,
        (Musl, Id::Sincosf) => 500,
        (Musl, Id::Tanh | Id::Tanhf) => 4,
        (Musl, Id::Tgamma) => 20,

        // Overrides for MPFR
        (Mpfr, Id::Acoshf) => 4,
        (Mpfr, Id::Asinh | Id::Asinhf) => 2,
        (Mpfr, Id::Atanh | Id::Atanhf) => 2,
        (Mpfr, Id::Exp10 | Id::Exp10f) => 3,
        (Mpfr, Id::Lgamma | Id::LgammaR | Id::Lgammaf | Id::LgammafR) => 16,
        (Mpfr, Id::Sinh | Id::Sinhf) => 2,
        (Mpfr, Id::Tanh | Id::Tanhf) => 2,
        (Mpfr, Id::Tgamma) => 20,

        // Defaults
        (Musl, _) => MUSL_DEFAULT_ULP,
        (Mpfr, _) => MP_DEFAULT_ULP,
    }
}

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
        input: (f32,),
        actual: F,
        expected: F,
        _ulp: &mut u32,
        ctx: &CheckCtx,
    ) -> Option<TestResult> {
        if ctx.basis == CheckBasis::Musl {
            if ctx.base_name == BaseName::Expm1 && input.0 > 80.0 && actual.is_infinite() {
                // we return infinity but the number is representable
                return XFAIL;
            }

            if ctx.base_name == BaseName::Sinh && input.0.abs() > 80.0 && actual.is_nan() {
                // we return some NaN that should be real values or infinite
                // doesn't seem to happen on x86
                return XFAIL;
            }
        }

        if ctx.base_name == BaseName::Acosh && input.0 < -1.0 {
            // acoshf is undefined for x <= 1.0, but we return a random result at lower
            // values.
            return XFAIL;
        }

        if ctx.base_name == BaseName::Lgamma || ctx.base_name == BaseName::LgammaR && input.0 < 0.0
        {
            // loggamma should not be defined for x < 0, yet we both return results
            return XFAIL;
        }

        maybe_check_nan_bits(actual, expected, ctx)
    }

    fn check_int<I: Int>(
        input: (f32,),
        actual: I,
        expected: I,
        ctx: &CheckCtx,
    ) -> Option<anyhow::Result<()>> {
        // On MPFR for lgammaf_r, we set -1 as the integer result for negative infinity but MPFR
        // sets +1
        if ctx.basis == CheckBasis::Mpfr
            && ctx.base_name == BaseName::LgammaR
            && input.0 == f32::NEG_INFINITY
            && actual.abs() == expected.abs()
        {
            XFAIL
        } else {
            None
        }
    }
}

impl MaybeOverride<(f64,)> for SpecialCase {
    fn check_float<F: Float>(
        input: (f64,),
        actual: F,
        expected: F,
        _ulp: &mut u32,
        ctx: &CheckCtx,
    ) -> Option<TestResult> {
        if ctx.basis == CheckBasis::Musl {
            if cfg!(target_arch = "x86") && ctx.base_name == BaseName::Acosh && input.0 < 1.0 {
                // The function is undefined, both implementations return random results
                return SKIP;
            }

            if cfg!(x86_no_sse)
                && ctx.base_name == BaseName::Ceil
                && input.0 < 0.0
                && input.0 > -1.0
                && expected == F::ZERO
                && actual == F::ZERO
            {
                // musl returns -0.0, we return +0.0
                return XFAIL;
            }
        }

        if ctx.base_name == BaseName::Acosh && input.0 < 1.0 {
            // The function is undefined for the inputs, musl and our libm both return
            // random results.
            return XFAIL;
        }

        if ctx.base_name == BaseName::Lgamma || ctx.base_name == BaseName::LgammaR && input.0 < 0.0
        {
            // loggamma should not be defined for x < 0, yet we both return results
            return XFAIL;
        }

        maybe_check_nan_bits(actual, expected, ctx)
    }

    fn check_int<I: Int>(
        input: (f64,),
        actual: I,
        expected: I,
        ctx: &CheckCtx,
    ) -> Option<anyhow::Result<()>> {
        // On MPFR for lgamma_r, we set -1 as the integer result for negative infinity but MPFR
        // sets +1
        if ctx.basis == CheckBasis::Mpfr
            && ctx.base_name == BaseName::LgammaR
            && input.0 == f64::NEG_INFINITY
            && actual.abs() == expected.abs()
        {
            XFAIL
        } else {
            None
        }
    }
}

/// Check NaN bits if the function requires it
fn maybe_check_nan_bits<F: Float>(actual: F, expected: F, ctx: &CheckCtx) -> Option<TestResult> {
    if !(ctx.base_name == BaseName::Fabs || ctx.base_name == BaseName::Copysign) {
        return None;
    }

    // LLVM currently uses x87 instructions which quieten signalling NaNs to handle the i686
    // `extern "C"` `f32`/`f64` return ABI.
    // LLVM issue <https://github.com/llvm/llvm-project/issues/66803>
    // Rust issue <https://github.com/rust-lang/rust/issues/115567>
    if cfg!(target_arch = "x86") && ctx.basis == CheckBasis::Musl {
        return SKIP;
    }

    // MPFR only has one NaN bitpattern; allow the default `.is_nan()` checks to validate.
    if ctx.basis == CheckBasis::Mpfr {
        return SKIP;
    }

    // abs and copysign require signaling NaNs to be propagated, so verify bit equality.
    if actual.to_bits() == expected.to_bits() {
        SKIP
    } else {
        Some(Err(anyhow::anyhow!("NaNs have different bitpatterns")))
    }
}

impl MaybeOverride<(f32, f32)> for SpecialCase {
    fn check_float<F: Float>(
        input: (f32, f32),
        _actual: F,
        expected: F,
        _ulp: &mut u32,
        ctx: &CheckCtx,
    ) -> Option<TestResult> {
        maybe_skip_binop_nan(input, expected, ctx)
    }
}

impl MaybeOverride<(f64, f64)> for SpecialCase {
    fn check_float<F: Float>(
        input: (f64, f64),
        _actual: F,
        expected: F,
        _ulp: &mut u32,
        ctx: &CheckCtx,
    ) -> Option<TestResult> {
        maybe_skip_binop_nan(input, expected, ctx)
    }
}

/// Musl propagates NaNs if one is provided as the input, but we return the other input.
// F1 and F2 are always the same type, this is just to please generics
fn maybe_skip_binop_nan<F1: Float, F2: Float>(
    input: (F1, F1),
    expected: F2,
    ctx: &CheckCtx,
) -> Option<TestResult> {
    match (&ctx.basis, ctx.base_name) {
        (Musl, BaseName::Fmin | BaseName::Fmax)
            if (input.0.is_nan() || input.1.is_nan()) && expected.is_nan() =>
        {
            XFAIL
        }

        (Mpfr, BaseName::Copysign) if input.1.is_nan() => SKIP,

        _ => None,
    }
}

impl MaybeOverride<(i32, f32)> for SpecialCase {
    fn check_float<F: Float>(
        input: (i32, f32),
        actual: F,
        expected: F,
        ulp: &mut u32,
        ctx: &CheckCtx,
    ) -> Option<TestResult> {
        match (&ctx.basis, ctx.base_name) {
            (Musl, _) => bessel_prec_dropoff(input, ulp, ctx),

            // We return +0.0, MPFR returns -0.0
            (Mpfr, BaseName::Jn)
                if input.1 == f32::NEG_INFINITY && actual == F::ZERO && expected == F::ZERO =>
            {
                XFAIL
            }

            _ => None,
        }
    }
}
impl MaybeOverride<(i32, f64)> for SpecialCase {
    fn check_float<F: Float>(
        input: (i32, f64),
        actual: F,
        expected: F,
        ulp: &mut u32,
        ctx: &CheckCtx,
    ) -> Option<TestResult> {
        match (&ctx.basis, ctx.base_name) {
            (Musl, _) => bessel_prec_dropoff(input, ulp, ctx),

            // We return +0.0, MPFR returns -0.0
            (Mpfr, BaseName::Jn)
                if input.1 == f64::NEG_INFINITY && actual == F::ZERO && expected == F::ZERO =>
            {
                XFAIL
            }

            _ => None,
        }
    }
}

/// Our bessel functions blow up with large N values
fn bessel_prec_dropoff<F: Float>(
    input: (i32, F),
    ulp: &mut u32,
    ctx: &CheckCtx,
) -> Option<TestResult> {
    if ctx.base_name == BaseName::Jn {
        if input.0 > 4000 {
            return XFAIL;
        } else if input.0 > 2000 {
            // *ulp = 20_000;
            *ulp = 20000;
        } else if input.0 > 1000 {
            *ulp = 4000;
        }
    }

    None
}

impl MaybeOverride<(f32, f32, f32)> for SpecialCase {}
impl MaybeOverride<(f64, f64, f64)> for SpecialCase {}
impl MaybeOverride<(f32, i32)> for SpecialCase {}
impl MaybeOverride<(f64, i32)> for SpecialCase {}
